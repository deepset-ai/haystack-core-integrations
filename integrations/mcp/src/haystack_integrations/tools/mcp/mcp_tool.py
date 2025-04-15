# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import threading
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, fields
from typing import Any, cast

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from haystack import logging
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool
from haystack.tools.errors import ToolInvocationError

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

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
            message = f"Failed to invoke tool '{tool_name}'"
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

    async def close(self) -> None:
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

    def close_sync(self) -> None:
        """Synchronous version of close for use in __del__ - ensures resources are cleaned up."""
        logger.debug("PROCESS: Closing StdioClient (sync)")

        try:
            AsyncExecutor.get_instance().run(self.close(), timeout=2)
        except Exception as e:
            logger.debug(f"PROCESS: Error during async cleanup in sync close: {e!s}")

    async def _initialize_session_with_transport(
        self,
        transport_tuple: tuple[
            MemoryObjectReceiveStream[types.JSONRPCMessage | Exception], MemoryObjectSendStream[types.JSONRPCMessage]
        ],
        connection_type: str,
    ) -> list[Tool]:
        """
        Common session initialization logic for all transports.

        :param transport_tuple: Tuple containing (stdio, write) from the transport
        :param connection_type: String describing the connection type for error messages
        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        try:
            self.stdio, self.write = transport_tuple
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            # Now session is guaranteed to be a ClientSession, not None
            session = cast(ClientSession, self.session)  # Tell mypy the type is now known
            await session.initialize()

            # List available tools
            response = await session.list_tools()
            return response.tools

        except Exception as e:
            await self.close()
            message = f"Failed to connect to {connection_type}: {e}"
            raise MCPConnectionError(message=message, operation="connect") from e


class StdioClient(MCPClient):
    """
    MCP client that connects to servers using stdio transport.
    """

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        """
        Initialize a stdio MCP client.

        :param command: Command to run (e.g., "python", "node")
        :param args: Arguments to pass to the command
        :param env: Environment variables for the command
        """
        super().__init__()
        self.command: str = command
        self.args: list[str] = args or []
        self.env: dict[str, str] | None = env
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

    def __init__(self, base_url: str, token: str | None = None, timeout: int = 5) -> None:
        """
        Initialize an SSE MCP client.

        :param base_url: Base URL of the server
        :param token: Authentication token for the server (optional)
        :param timeout: Connection timeout in seconds
        """
        super().__init__()
        self.base_url: str = base_url.rstrip("/")  # Remove any trailing slashes
        self.token: str | None = token
        self.timeout: int = timeout

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using SSE transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        sse_url = f"{self.base_url}/sse"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(sse_url, headers=headers, timeout=self.timeout)
        )
        return await self._initialize_session_with_transport(sse_transport, f"HTTP server at {self.base_url}")


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
        for field in fields(self):
            result[field.name] = getattr(self, field.name)

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

        # Create an instance of the class with the remaining fields
        return cls(**data_copy)


@dataclass
class SSEServerInfo(MCPServerInfo):
    """
    Data class that encapsulates SSE MCP server connection parameters.

    :param base_url: Base URL of the MCP server
    :param token: Authentication token for the server (optional)
    :param timeout: Connection timeout in seconds
    """

    base_url: str
    token: str | None = None
    timeout: int = 30

    def create_client(self) -> MCPClient:
        """
        Create an SSE MCP client.

        :returns: Configured HttpMCPClient instance
        """
        return SSEClient(self.base_url, self.token, self.timeout)


@dataclass
class StdioServerInfo(MCPServerInfo):
    """
    Data class that encapsulates stdio MCP server connection parameters.

    :param command: Command to run (e.g., "python", "node")
    :param args: Arguments to pass to the command
    :param env: Environment variables for the command
    """

    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None

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
        server_info=SSEServerInfo(base_url="http://localhost:8000")
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

        # Create client
        self._client = server_info.create_client()
        logger.debug(f"TOOL: Created client for MCPTool '{name}'")

        try:

            async def connect():
                logger.debug(f"TOOL: Inside connect coroutine for '{name}'")
                result = await asyncio.wait_for(self._client.connect(), timeout=connection_timeout)
                logger.debug(f"TOOL: Connect successful for '{name}', found {len(result)} tools")
                return result

            logger.debug(f"TOOL: About to run connect for '{name}'")
            tools = AsyncExecutor.get_instance().run(connect(), timeout=connection_timeout)
            logger.debug(f"TOOL: Connection complete for '{name}'")

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
            # Clean up resources on error
            logger.debug(f"TOOL: Error during initialization of '{name}': {e!s}")
            if self._client:
                try:
                    logger.debug(f"TOOL: Attempting cleanup after initialization failure for '{name}'")
                    AsyncExecutor.get_instance().run(self._client.close(), timeout=5)
                    logger.debug(f"TOOL: Cleanup successful for '{name}'")
                except Exception as cleanup_error:
                    logger.debug(f"TOOL: Error during cleanup after initialization failure: {cleanup_error!s}")

            message = f"Failed to initialize MCPTool '{name}': {e}"
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
            message = f"Failed to invoke tool '{self.name}'"
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
            message = f"Failed to invoke tool '{self.name}'"
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

    def __del__(self):
        """Cleanup resources when the tool is garbage collected."""
        logger.debug(f"TOOL: __del__ called for MCPTool '{self.name if hasattr(self, 'name') else 'unknown'}'")

        # Call synchronous close on the client
        if hasattr(self, "_client") and self._client:
            try:
                self._client.close_sync()
            except Exception as e:
                logger.debug(f"TOOL: Error during synchronous client close: {e!s}")
