# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, cast

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from haystack import logging
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool
from haystack.tools.errors import ToolInvocationError
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.auth import SecretType
from haystack.utils.url_validation import is_valid_http_url

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    streamablehttp_client = None

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """Thread-safe event loop executor for running async code from sync contexts."""

    _singleton_instance = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AsyncExecutor":
        """Get or create the global singleton executor instance."""
        with cls._singleton_lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = cls()
            return cls._singleton_instance

    def __init__(self):
        """Initialize a dedicated event loop"""
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(target=self._run_loop, daemon=True)
        self._started = threading.Event()
        self._thread.start()
        if not self._started.wait(timeout=5):
            message = "AsyncExecutor failed to start background event loop"
            raise RuntimeError(message)

    def _run_loop(self):
        """Run the event loop"""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """
        Run a coroutine in the event loop.

        :param coro: Coroutine to execute
        :param timeout: Optional timeout in seconds
        :return: Result of the coroutine
        :raises TimeoutError: If execution exceeds timeout
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            message = f"Operation timed out after {timeout} seconds"
            raise TimeoutError(message) from e

    def get_loop(self):
        """
        Get the event loop.

        :returns: The event loop
        """
        return self._loop

    def run_background(
        self, coro_factory: Callable[[asyncio.Event], Coroutine[Any, Any, Any]], timeout: float | None = None
    ) -> tuple[concurrent.futures.Future[Any], asyncio.Event]:
        """
        Schedule `coro_factory` to run in the executor's event loop **without** blocking the
        caller thread.

        The factory receives an :class:`asyncio.Event` that can be used to cooperatively shut
        the coroutine down. The method returns **both** the concurrent future (to observe
        completion or failure) and the created *stop_event* so that callers can signal termination.

        :param coro_factory: A callable receiving the stop_event and returning the coroutine to execute.
        :param timeout: Optional timeout while waiting for the stop_event to be created.
        :returns: Tuple ``(future, stop_event)``.
        """
        # A promise that will be fulfilled from inside the coroutine_with_stop_event coroutine once the
        # stop_event is created *inside* the target event loop to ensure it is bound to the
        # correct loop and can safely be set from other threads via *call_soon_threadsafe*.
        stop_event_promise: Future[asyncio.Event] = Future()

        async def _coroutine_with_stop_event():
            stop_event = asyncio.Event()
            stop_event_promise.set_result(stop_event)
            await coro_factory(stop_event)

        # Schedule the coroutine
        future = asyncio.run_coroutine_threadsafe(_coroutine_with_stop_event(), self._loop)

        # This ensures that the stop_event is fully initialized and ready for use before
        # the run_background method returns, allowing the caller to immediately
        # use it to control the coroutine.
        return future, stop_event_promise.result(timeout)

    def shutdown(self, timeout: float = 2):
        """
        Shut down the background event loop and thread.

        :param timeout: Timeout in seconds for shutting down the event loop
        """

        def _stop_loop():
            self._loop.stop()

        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result(timeout=timeout)
        self._loop.call_soon_threadsafe(_stop_loop)
        self._thread.join(timeout=timeout)


class MCPError(Exception):
    """Base class for MCP-related errors."""

    def __init__(self, message: str) -> None:
        """
        Initialize the MCPError.

        :param message: Descriptive error message
        """
        super().__init__(message)
        self.message = message


class MCPConnectionError(MCPError):
    """Error connecting to MCP server."""

    def __init__(self, message: str, server_info: "MCPServerInfo | None" = None, operation: str | None = None) -> None:
        """
        Initialize the MCPConnectionError.

        :param message: Descriptive error message
        :param server_info: Server connection information that was used
        :param operation: Name of the operation that was being attempted
        """
        super().__init__(message)
        self.server_info = server_info
        self.operation = operation


class MCPToolNotFoundError(MCPError):
    """Error when a tool is not found on the server."""

    def __init__(self, message: str, tool_name: str, available_tools: list[str] | None = None) -> None:
        """
        Initialize the MCPToolNotFoundError.

        :param message: Descriptive error message
        :param tool_name: Name of the tool that was requested but not found
        :param available_tools: List of available tool names, if known
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class MCPResponseTypeError(MCPError):
    """Error when response content type is not supported."""

    def __init__(self, message: str, response: Any, tool_name: str | None = None) -> None:
        """
        Initialize the MCPResponseTypeError.

        :param message: Descriptive error message
        :param response: The response that had the wrong type
        :param tool_name: Name of the tool that produced the response
        """
        super().__init__(message)
        self.response = response
        self.tool_name = tool_name


class MCPInvocationError(ToolInvocationError):
    """Error during tool invocation."""

    def __init__(self, message: str, tool_name: str, tool_args: dict[str, Any] | None = None) -> None:
        """
        Initialize the MCPInvocationError.

        :param message: Descriptive error message
        :param tool_name: Name of the tool that was being invoked
        :param tool_args: Arguments that were passed to the tool
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}


class MCPClient(ABC):
    """
    Abstract base class for MCP clients.

    This class defines the common interface and shared functionality for all MCP clients,
    regardless of the transport mechanism used.
    """

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.stdio: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception] | None = None
        self.write: MemoryObjectSendStream[types.JSONRPCMessage] | None = None

    @abstractmethod
    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        pass

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """
        Call a tool on the connected MCP server.

        :param tool_name: Name of the tool to call
        :param tool_args: Arguments to pass to the tool
        :returns: Result of the tool invocation
        :raises MCPConnectionError: If not connected to an MCP server
        :raises MCPInvocationError: If the tool invocation fails
        :raises MCPResponseTypeError: If response type is not TextContent
        """
        if not self.session:
            message = "Not connected to an MCP server"
            raise MCPConnectionError(message=message, operation="call_tool")

        try:
            result = await self.session.call_tool(tool_name, tool_args)
            validated_result = self._validate_response(tool_name, result)
            return validated_result
        except MCPError:
            # Re-raise specific MCP errors directly
            raise
        except Exception as e:
            # Wrap other exceptions with context about which tool failed
            message = f"Failed to invoke tool '{tool_name}' with args: {tool_args} , got error: {e!s}"
            raise MCPInvocationError(message, tool_name, tool_args) from e

    def _validate_response(self, tool_name: str, result: types.CallToolResult) -> types.CallToolResult:
        """
        Validate response from an MCP tool call, accepting only TextContent.

        :param tool_name: Name of the called tool (for error messages)
        :param result: CallToolResult from MCP tool call
        :returns: The original CallToolResult object
        :raises MCPResponseTypeError: If content type is not TextContent
        :raises MCPInvocationError: If the tool call resulted in an error
        """

        # Check for error response
        if result.isError:
            if len(result.content) > 0 and isinstance(result.content[0], types.TextContent):
                # Get the error message from the first item
                first_item = result.content[0]
                message = f"Tool '{tool_name}' returned an error: {first_item.text}"
            else:
                message = f"Tool '{tool_name}' returned an error: {result.content!s}"
            raise MCPInvocationError(
                message=message,
                tool_name=tool_name,
            )

        # Validate content types - only allow TextContent for now
        if result.content:
            for item in result.content:
                if not isinstance(item, types.TextContent):
                    # Reject any non-TextContent
                    message = (
                        f"Unsupported content type in response from tool '{tool_name}'. "
                        f"Only TextContent is currently supported."
                    )
                    raise MCPResponseTypeError(message, result, tool_name)

        # Return the original result object
        return result

    async def aclose(self) -> None:
        """
        Close the connection and clean up resources.

        This method ensures all resources are properly released, even if errors occur.
        """
        if not self.exit_stack:
            return

        try:
            await self.exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error during MCP client cleanup: {e}")
        finally:
            # Ensure all references are cleared even if cleanup fails
            self.session = None
            self.stdio = None
            self.write = None

    async def _initialize_session_with_transport(
        self,
        transport_tuple: tuple[
            MemoryObjectReceiveStream[types.JSONRPCMessage | Exception], MemoryObjectSendStream[types.JSONRPCMessage]
        ],
        connection_type: str,
    ) -> list[Tool]:
        """
        Common session initialization logic for all transports.

        :param transport_tuple: Tuple containing at least (stdio, write) from the transport
        :param connection_type: String describing the connection type for error messages
        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        try:
            # Use extended unpacking to handle tuples with additional values
            self.stdio, self.write, *_ = transport_tuple
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            # Now session is guaranteed to be a ClientSession, not None
            session = cast(ClientSession, self.session)  # Tell mypy the type is now known
            await session.initialize()

            # List available tools
            response = await session.list_tools()
            return response.tools

        except Exception as e:
            # We'll clean up the session in the calling code, so we don't need to do it here.
            message = f"Failed to connect to {connection_type}: {e}"
            raise MCPConnectionError(message=message, operation="connect") from e


class StdioClient(MCPClient):
    """
    MCP client that connects to servers using stdio transport.
    """

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str | Secret] | None = None) -> None:
        """
        Initialize a stdio MCP client.

        :param command: Command to run (e.g., "python", "node")
        :param args: Arguments to pass to the command
        :param env: Environment variables for the command
        """
        super().__init__()
        self.command: str = command
        self.args: list[str] = args or []
        # Resolve Secret values in environment variables
        self.env: dict[str, str] | None = None
        if env:
            self.env = {
                key: value.resolve_value() if isinstance(value, Secret) else value for key, value in env.items()
            }
        logger.debug(f"PROCESS: Created StdioClient for command: {command} {' '.join(self.args or [])}")

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using stdio transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        logger.debug(f"PROCESS: Connecting to stdio server with command: {self.command}")

        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        return await self._initialize_session_with_transport(stdio_transport, f"stdio server (command: {self.command})")


class SSEClient(MCPClient):
    """
    MCP client that connects to servers using SSE transport.
    """

    def __init__(self, server_info: "SSEServerInfo") -> None:
        """
        Initialize an SSE MCP client using server configuration.

        :param server_info: Configuration object containing URL, token, timeout, etc.
        """
        super().__init__()

        # in post_init we validate the url and set the url field so it is guaranteed to be valid
        # safely ignore the mypy warning here
        self.url: str = server_info.url  # type: ignore[assignment]
        self.token: str | None = (
            server_info.token.resolve_value() if isinstance(server_info.token, Secret) else server_info.token
        )
        self.timeout: int = server_info.timeout

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using SSE transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(self.url, headers=headers, timeout=self.timeout)
        )
        return await self._initialize_session_with_transport(sse_transport, f"HTTP server at {self.url}")


class StreamableHttpClient(MCPClient):
    """
    MCP client that connects to servers using streamable HTTP transport.
    """

    def __init__(self, server_info: "StreamableHttpServerInfo") -> None:
        """
        Initialize a streamable HTTP MCP client using server configuration.

        :param server_info: Configuration object containing URL, token, timeout, etc.
        """
        super().__init__()

        self.url: str = server_info.url
        self.token: str | None = (
            server_info.token.resolve_value() if isinstance(server_info.token, Secret) else server_info.token
        )
        self.timeout: int = server_info.timeout

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using streamable HTTP transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        if streamablehttp_client is None:
            message = (
                "Streamable HTTP client is not available. "
                "This may require a newer version of the mcp package that includes mcp.client.streamable_http"
            )
            raise MCPConnectionError(message=message, operation="streamable_http_connect")

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        streamablehttp_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url=self.url, headers=headers, timeout=timedelta(seconds=self.timeout))
        )
        return await self._initialize_session_with_transport(streamablehttp_transport, f"HTTP server at {self.url}")


@dataclass
class MCPServerInfo(ABC):
    """
    Abstract base class for MCP server connection parameters.

    This class defines the common interface for all MCP server connection types.
    """

    @abstractmethod
    def create_client(self) -> MCPClient:
        """
        Create an appropriate MCP client for this server info.

        :returns: An instance of MCPClient configured with this server info
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this server info to a dictionary.

        :returns: Dictionary representation of this server info
        """
        # Store the fully qualified class name for deserialization
        result = {"type": generate_qualified_class_name(type(self))}

        # Add all fields from the dataclass
        for dataclass_field in fields(self):
            value = getattr(self, dataclass_field.name)
            if isinstance(value, Secret):
                result[dataclass_field.name] = value.to_dict()
            elif isinstance(value, dict):
                # Handle dicts that may contain Secret objects
                serialized_dict = {}
                for k, v in value.items():
                    if isinstance(v, Secret):
                        serialized_dict[k] = v.to_dict()
                    else:
                        serialized_dict[k] = v
                result[dataclass_field.name] = serialized_dict
            else:
                result[dataclass_field.name] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerInfo":
        """
        Deserialize server info from a dictionary.

        :param data: Dictionary containing serialized server info
        :returns: Instance of the appropriate server info class
        """
        # Remove the type field as it's not a constructor parameter
        data_copy = data.copy()
        data_copy.pop("type", None)

        # Handle Secret deserialization for any field
        for dataclass_field in fields(cls):
            if dataclass_field.name in data_copy:
                field_value = data_copy[dataclass_field.name]
                if isinstance(field_value, dict):
                    if "type" in field_value and field_value["type"] in [e.value for e in SecretType]:
                        # The whole field is a Secret e.g. token field of SSEServerInfo
                        deserialize_secrets_inplace(data_copy, keys=[dataclass_field.name])
                    else:
                        # Most likely env field of StdioServerInfo
                        for key, value in field_value.items():
                            if (
                                isinstance(value, dict)
                                and "type" in value
                                and value["type"] in [e.value for e in SecretType]
                            ):
                                deserialize_secrets_inplace(field_value, keys=[key])

        # Create an instance of the class with the remaining fields
        return cls(**data_copy)


@dataclass
class SSEServerInfo(MCPServerInfo):
    """
    Data class that encapsulates SSE MCP server connection parameters.

    For authentication tokens containing sensitive data, you can use Secret objects
    for secure handling and serialization:

    ```python
    server_info = SSEServerInfo(
        url="https://my-mcp-server.com",
        token=Secret.from_env_var("API_KEY"),
    )
    ```

    :param url: Full URL of the MCP server (including /sse endpoint)
    :param base_url: Base URL of the MCP server (deprecated, use url instead)
    :param token: Authentication token for the server (optional)
    :param timeout: Connection timeout in seconds
    """

    url: str | None = None
    base_url: str | None = None  # deprecated
    token: str | Secret | None = None
    timeout: int = 30

    def __post_init__(self):
        """Validate that either url or base_url is provided."""
        if not self.url and not self.base_url:
            message = "Either url or base_url must be provided"
            raise ValueError(message)
        if self.url and self.base_url:
            message = "Only one of url or base_url should be provided, if both are provided, base_url will be ignored"
            warnings.warn(message, DeprecationWarning, stacklevel=2)

        if self.base_url:
            if not is_valid_http_url(self.base_url):
                message = f"Invalid base_url: {self.base_url}"
                raise ValueError(message)

            warnings.warn(
                "base_url is deprecated and will be removed in a future version. Use url instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # from now on only use url for the lifetime of the SSEServerInfo instance, never base_url
            self.url = f"{self.base_url.rstrip('/')}/sse"

        elif not is_valid_http_url(self.url):
            message = f"Invalid url: {self.url}"
            raise ValueError(message)

    def create_client(self) -> MCPClient:
        """
        Create an SSE MCP client.

        :returns: Configured MCPClient instance
        """
        # Pass the validated SSEServerInfo instance directly
        return SSEClient(server_info=self)


@dataclass
class StreamableHttpServerInfo(MCPServerInfo):
    """
    Data class that encapsulates streamable HTTP MCP server connection parameters.

    For authentication tokens containing sensitive data, you can use Secret objects
    for secure handling and serialization:

    ```python
    server_info = StreamableHttpServerInfo(
        url="https://my-mcp-server.com",
        token=Secret.from_env_var("API_KEY"),
    )
    ```

    :param url: Full URL of the MCP server (streamable HTTP endpoint)
    :param token: Authentication token for the server (optional)
    :param timeout: Connection timeout in seconds
    """

    url: str
    token: str | Secret | None = None
    timeout: int = 30

    def __post_init__(self):
        """Validate the URL."""
        if not is_valid_http_url(self.url):
            message = f"Invalid url: {self.url}"
            raise ValueError(message)

    def create_client(self) -> MCPClient:
        """
        Create a streamable HTTP MCP client.

        :returns: Configured StreamableHttpClient instance
        """
        return StreamableHttpClient(server_info=self)


@dataclass
class StdioServerInfo(MCPServerInfo):
    """
    Data class that encapsulates stdio MCP server connection parameters.

    :param command: Command to run (e.g., "python", "node")
    :param args: Arguments to pass to the command
    :param env: Environment variables for the command

    For environment variables containing sensitive data, you can use Secret objects
    for secure handling and serialization:

    ```python
    server_info = StdioServerInfo(
        command="uv",
        args=["run", "my-mcp-server"],
        env={
            "WORKSPACE_PATH": "/path/to/workspace",  # Plain string
            "API_KEY": Secret.from_env_var("API_KEY"),  # Secret object
        }
    )
    ```

    Secret objects will be properly serialized and deserialized without exposing
    the secret value, while plain strings will be preserved as-is. Use Secret objects
    for sensitive data that needs to be handled securely.
    """

    command: str
    args: list[str] | None = None
    env: dict[str, str | Secret] | None = None

    def create_client(self) -> MCPClient:
        """
        Create a stdio MCP client.

        :returns: Configured StdioMCPClient instance
        """
        return StdioClient(self.command, self.args, self.env)


class MCPTool(Tool):
    """
    A Tool that represents a single tool from an MCP server.

    This implementation uses the official MCP SDK for protocol handling while maintaining
    compatibility with the Haystack tool ecosystem.

    Response handling:
    - Text content is supported and returned as strings
    - Unsupported content types (like binary/images) will raise MCPResponseTypeError

    Example using HTTP:
    ```python
    from haystack.tools import MCPTool, SSEServerInfo

    # Create tool instance
    tool = MCPTool(
        name="add",
        server_info=SSEServerInfo(url="http://localhost:8000/sse")
    )

    # Use the tool
    result = tool.invoke(a=5, b=3)
    ```

    Example using stdio:
    ```python
    from haystack.tools import MCPTool, StdioServerInfo

    # Create tool instance
    tool = MCPTool(
        name="get_current_time",
        server_info=StdioServerInfo(command="python", args=["path/to/server.py"])
    )

    # Use the tool
    result = tool.invoke(timezone="America/New_York")
    ```
    """

    def __init__(
        self,
        name: str,
        server_info: MCPServerInfo,
        description: str | None = None,
        connection_timeout: int = 30,
        invocation_timeout: int = 30,
    ):
        """
        Initialize the MCP tool.

        :param name: Name of the tool to use
        :param server_info: Server connection information
        :param description: Custom description (if None, server description will be used)
        :param connection_timeout: Timeout in seconds for server connection
        :param invocation_timeout: Default timeout in seconds for tool invocations
        :raises MCPConnectionError: If connection to the server fails
        :raises MCPToolNotFoundError: If no tools are available or the requested tool is not found
        :raises TimeoutError: If connection times out
        """

        # Store connection parameters for serialization
        self._server_info = server_info
        self._connection_timeout = connection_timeout
        self._invocation_timeout = invocation_timeout

        logger.debug(f"TOOL: Initializing MCPTool '{name}'")

        try:
            # Create client and spin up a long-lived worker that keeps the
            # connect/close lifecycle inside one coroutine.
            self._client = server_info.create_client()
            logger.debug(f"TOOL: Created client for MCPTool '{name}'")

            # The worker starts immediately and blocks here until the connection
            # is established (or fails), returning the tool list.
            self._worker = _MCPClientSessionManager(self._client, timeout=connection_timeout)

            tools = self._worker.tools()
            # Handle no tools case
            if not tools:
                logger.debug(f"TOOL: No tools found for '{name}'")
                message = "No tools available on server"
                raise MCPToolNotFoundError(message, tool_name=name)

            # Find the specified tool
            tool_dict = {t.name: t for t in tools}
            logger.debug(f"TOOL: Available tools: {list(tool_dict.keys())}")

            tool_info = tool_dict.get(name)

            if not tool_info:
                available = list(tool_dict.keys())
                logger.debug(f"TOOL: Tool '{name}' not found in available tools")
                message = f"Tool '{name}' not found on server. Available tools: {', '.join(available)}"
                raise MCPToolNotFoundError(message, tool_name=name, available_tools=available)

            logger.debug(f"TOOL: Found tool '{name}', initializing Tool parent class")
            # Initialize the parent class
            super().__init__(
                name=name,
                description=description or tool_info.description,
                parameters=tool_info.inputSchema,
                function=self._invoke_tool,
            )
            logger.debug(f"TOOL: Initialization complete for '{name}'")

        except Exception as e:
            # We need to close because we could connect properly, retrieve tools yet
            # fail because of an MCPToolNotFoundError
            self.close()

            # Extract more detailed error information from TaskGroup/ExceptionGroup exceptions
            from exceptiongroup import ExceptionGroup

            error_message = str(e)
            # Handle ExceptionGroup to extract more useful error messages
            if isinstance(e, ExceptionGroup):
                if e.exceptions:
                    first_exception = e.exceptions[0]
                    error_message = (
                        first_exception.message if hasattr(first_exception, "message") else str(first_exception)
                    )

            # Ensure we always have a meaningful error message
            if not error_message or error_message.strip() == "":
                # Provide platform-independent fallback message for connection errors
                error_message = f"Connection failed to MCP server (using {type(server_info).__name__})"

            message = f"Failed to initialize MCPTool '{name}': {error_message}"
            raise MCPConnectionError(message=message, server_info=server_info, operation="initialize") from e

    def _invoke_tool(self, **kwargs: Any) -> Any:
        """
        Synchronous tool invocation.

        :param kwargs: Arguments to pass to the tool
        :returns: Result of the tool invocation
        """
        logger.debug(f"TOOL: Invoking tool '{self.name}' with args: {kwargs}")
        try:

            async def invoke():
                logger.debug(f"TOOL: Inside invoke coroutine for '{self.name}'")
                result = await asyncio.wait_for(
                    self._client.call_tool(self.name, kwargs), timeout=self._invocation_timeout
                )
                logger.debug(f"TOOL: Invoke successful for '{self.name}'")
                return result

            logger.debug(f"TOOL: About to run invoke for '{self.name}'")
            result = AsyncExecutor.get_instance().run(invoke(), timeout=self._invocation_timeout)
            logger.debug(f"TOOL: Invoke complete for '{self.name}', result type: {type(result)}")
            return result
        except (MCPError, TimeoutError) as e:
            logger.debug(f"TOOL: Known error during invoke of '{self.name}': {e!s}")
            # Pass through known errors
            raise
        except Exception as e:
            # Wrap other errors
            logger.debug(f"TOOL: Unknown error during invoke of '{self.name}': {e!s}")
            message = f"Failed to invoke tool '{self.name}' with args: {kwargs} , got error: {e!s}"
            raise MCPInvocationError(message, self.name, kwargs) from e

    async def ainvoke(self, **kwargs: Any) -> Any:
        """
        Asynchronous tool invocation.

        :param kwargs: Arguments to pass to the tool
        :returns: Result of the tool invocation, processed based on content type
        :raises MCPInvocationError: If the tool invocation fails
        :raises MCPResponseTypeError: If response type is not supported
        :raises TimeoutError: If the operation times out
        """
        try:
            return await asyncio.wait_for(self._client.call_tool(self.name, kwargs), timeout=self._invocation_timeout)
        except asyncio.TimeoutError as e:
            message = f"Tool invocation timed out after {self._invocation_timeout} seconds"
            raise TimeoutError(message) from e
        except Exception as e:
            if isinstance(e, MCPError):
                raise
            message = f"Failed to invoke tool '{self.name}' with args: {kwargs} , got error: {e!s}"
            raise MCPInvocationError(message, self.name, kwargs) from e

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the MCPTool to a dictionary.

        The serialization preserves all information needed to recreate the tool,
        including server connection parameters and timeout settings. Note that the
        active connection is not maintained.

        :returns: Dictionary with serialized data in the format:
                  {"type": fully_qualified_class_name, "data": {parameters}}
        """
        serialized = {
            "name": self.name,
            "description": self.description,
            "server_info": self._server_info.to_dict(),
            "connection_timeout": self._connection_timeout,
            "invocation_timeout": self._invocation_timeout,
        }
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": serialized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tool":
        """
        Deserializes the MCPTool from a dictionary.

        This method reconstructs an MCPTool instance from a serialized dictionary,
        including recreating the server_info object. A new connection will be established
        to the MCP server during initialization.

        :param data: Dictionary containing serialized tool data
        :returns: A fully initialized MCPTool instance
        :raises: Various exceptions if connection fails
        """
        # Extract the tool parameters from the data dictionary
        inner_data = data["data"]
        server_info_dict = inner_data.get("server_info", {})

        # Reconstruct the server_info object
        # First get the appropriate class by name
        server_info_class = import_class_by_name(server_info_dict["type"])
        # Then deserialize using that class's from_dict method
        server_info = server_info_class.from_dict(server_info_dict)

        # Handle backward compatibility for timeout parameters
        connection_timeout = inner_data.get("connection_timeout", 30)
        invocation_timeout = inner_data.get("invocation_timeout", 30)

        # Create a new MCPTool instance with the deserialized parameters
        # This will establish a new connection to the MCP server
        return cls(
            name=inner_data["name"],
            description=inner_data.get("description"),
            server_info=server_info,
            connection_timeout=connection_timeout,
            invocation_timeout=invocation_timeout,
        )

    def close(self):
        """Close the tool synchronously."""
        if hasattr(self, "_client") and self._client:
            try:
                # Tell the background worker to shut down gracefully.
                if hasattr(self, "_worker") and self._worker:
                    self._worker.stop()
            except Exception as e:
                logger.debug(f"TOOL: Error during synchronous worker stop: {e!s}")

    def __del__(self):
        """Cleanup resources when the tool is garbage collected."""
        logger.debug(f"TOOL: __del__ called for MCPTool '{self.name if hasattr(self, 'name') else 'unknown'}'")

        self.close()


class _MCPClientSessionManager:
    """Runs an MCPClient connect/close inside the AsyncExecutor's event loop.

    Life-cycle:
      1.  Create the worker to schedule a long-running coroutine in the
          dedicated background loop.
      2.  The coroutine calls *connect* on mcp client; when it has the tool list it fulfils
          a concurrent future so the synchronous thread can continue.
      3.  It then waits on an `asyncio.Event`.
      4.  `stop()` sets the event from any thread. The same coroutine then calls
          *close()* on mcp client and finishes without the dreaded
          `Attempted to exit cancel scope in a different task than it was entered in` error
          thus properly closing the client.
    """

    # Maximum time to wait for worker shutdown in seconds
    WORKER_SHUTDOWN_TIMEOUT = 2.0

    def __init__(self, client: "MCPClient", *, timeout: float | None = None):
        self._client = client
        self.executor = AsyncExecutor.get_instance()

        # Where the tool list (or an exception) will be delivered.
        self._tools_promise: Future[list[Tool]] = Future()

        # Kick off the worker coroutine in the background loop
        self._worker_future, self._stop_event = self.executor.run_background(self._run, timeout=None)

        # Wait (in the caller thread) until connect() finishes or raises.
        try:
            self._tools_promise.result(timeout)
        except BaseException:
            # If connect failed we should cancel the worker so it doesn't hang.
            self.stop()
            raise

    def tools(self) -> list[Tool]:
        """Return the tool list already collected during startup."""

        return self._tools_promise.result()

    def stop(self) -> None:
        """Request the worker to shut down and block until done."""

        def _set(ev: asyncio.Event):
            if not ev.is_set():
                ev.set()

        if self.executor.get_loop().is_closed():
            return

        # The stop event is created inside the worker *before* the connect
        # promise is fulfilled, so at this point it must exist.
        self.executor.get_loop().call_soon_threadsafe(_set, self._stop_event)  # type: ignore[attr-defined]

        # Wait for the worker coroutine to finish so resources are fully
        # released before returning. Swallow any errors during shutdown.
        try:
            self._worker_future.result(timeout=self.WORKER_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.debug(f"Error during worker future result: {e}")
            pass

    async def _run(self, stop_event: asyncio.Event):
        """Background coroutine living in AsyncExecutor's loop."""

        try:
            # logger.debug(f"TOOL: _run current task: {asyncio.current_task()}")
            tools = await self._client.connect()
            # Deliver the tool list to the waiting synchronous code.
            if not self._tools_promise.done():
                self._tools_promise.set_result(tools)
            # Park until told to stop.
            await stop_event.wait()
        except Exception as exc:
            logger.debug(f"Error during _run: {exc}")
            if not self._tools_promise.done():
                self._tools_promise.set_exception(exc)
            raise
        finally:
            # logger.debug(f"TOOL: _run current task: {asyncio.current_task()}")
            # Close the client in the same couroutine that connected it
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug(f"Error during MCP client cleanup: {e!s}")
