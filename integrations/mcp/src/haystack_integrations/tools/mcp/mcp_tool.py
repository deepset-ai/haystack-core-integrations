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

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from exceptiongroup import ExceptionGroup
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
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)


def _resolve_headers(headers: dict[str, str | Secret] | None) -> dict[str, str] | None:
    """
    Resolve Secret values in headers dictionary and warn about None values.

    :param headers: Dictionary of headers, potentially containing Secret objects
    :returns: Dictionary with resolved string values, or None if input is None
    """
    if not headers:
        return None

    resolved_headers = {}
    for key, value in headers.items():
        resolved_value = value.resolve_value() if isinstance(value, Secret) else value
        if resolved_value is None:
            logger.warning(
                f"Header '{key}' resolved to None. This may indicate a misconfiguration. "
                f"The header will be set to an empty string."
            )
            resolved_headers[key] = ""
        else:
            resolved_headers[key] = resolved_value

    return resolved_headers


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

    def shutdown(self, timeout: float = 2) -> None:
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


class MCPInvocationError(ToolInvocationError):
    """Error during tool invocation."""

    def __init__(self, message: str, tool_name: str, tool_args: dict[str, Any] | None = None) -> None:
        """
        Initialize the MCPInvocationError.

        :param message: Descriptive error message
        :param tool_name: Name of the tool that was being invoked
        :param tool_args: Arguments that were passed to the tool
        """
        super().__init__(message=message, tool_name=tool_name)
        self.tool_args = tool_args or {}


class MCPClient(ABC):
    """
    Abstract base class for MCP clients.

    This class defines the common interface and shared functionality for all MCP clients,
    regardless of the transport mechanism used.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0) -> None:
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.stdio: MemoryObjectReceiveStream[SessionMessage | Exception] | None = None
        self.write: MemoryObjectSendStream[SessionMessage] | None = None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    @abstractmethod
    async def connect(self) -> list[types.Tool]:
        """
        Connect to an MCP server.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        pass

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """
        Call a tool on the connected MCP server.

        :param tool_name: Name of the tool to call
        :param tool_args: Arguments to pass to the tool
        :returns: JSON string representation of the tool invocation result
        :raises MCPConnectionError: If not connected to an MCP server
        :raises MCPInvocationError: If the tool invocation fails
        """
        if not self.session:
            message = "Not connected to an MCP server"
            raise MCPConnectionError(message=message, operation="call_tool")

        # Implement retry logic with exponential backoff for connection-related errors
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                result = await self.session.call_tool(tool_name, tool_args)
                if result.isError:
                    message = f"Tool '{tool_name}' returned an error: {result.content!s}"
                    raise MCPInvocationError(message=message, tool_name=tool_name)
                return result.model_dump_json()
            except MCPError:
                # Re-raise specific MCP errors directly (these are not connection issues)
                raise
            except (anyio.ClosedResourceError, ConnectionError, OSError) as e:
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else "Connection closed unexpectedly"

                # Don't retry on the last attempt
                if attempt >= self.max_retries:
                    message = (
                        f"Tool '{tool_name}' failed after {self.max_retries} reconnection attempts. "
                        f"Last error: {error_type}: {error_msg}. "
                        f"Consider recreating the MCPTool instance or checking server availability."
                    )
                    raise MCPInvocationError(message, tool_name, tool_args) from e

                # Only attempt reconnection for SSE/HTTP transports (if available)
                if isinstance(self, SSEClient | StreamableHttpClient) and (
                    sse_client is not None or streamablehttp_client is not None
                ):
                    logger.warning(f"Connection lost during tool call '{tool_name}': {error_type}: {error_msg}")

                    try:
                        # Exponential backoff before reconnection attempt
                        if attempt > 0:
                            delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                            logger.info(f"Waiting {delay}s before reconnection attempt {attempt + 1}")
                            await asyncio.sleep(delay)

                        logger.info(f"Attempting reconnection {attempt + 1}/{self.max_retries}")
                        await self.connect()
                        logger.info(f"Reconnection {attempt + 1} successful")
                        # Continue to next iteration to retry the tool call

                    except Exception as reconnect_error:
                        logger.error(f"Reconnection attempt {attempt + 1} failed: {reconnect_error}")
                        if attempt >= self.max_retries - 1:  # This was the last reconnection attempt
                            message = (
                                f"Tool '{tool_name}' failed and all {self.max_retries} reconnection attempts failed. "
                                f"Original error: {error_type}: {error_msg}. "
                                f"Final reconnection error: {reconnect_error}. "
                                f"Try recreating the MCPTool instance."
                            )
                            raise MCPInvocationError(message, tool_name, tool_args) from e
                        # Continue to next iteration for another reconnection attempt
                else:
                    # For non-HTTP transports, don't attempt reconnection
                    message = (
                        f"Connection lost during tool call '{tool_name}': {error_type}: {error_msg}. "
                        f"For STDIO connections, try recreating the MCPTool instance."
                    )
                    raise MCPInvocationError(message, tool_name, tool_args) from e
            except Exception as e:
                # Handle other exceptions with meaningful error messages
                error_description = str(e) if str(e) else f"Unknown {type(e).__name__} error"
                message = f"Failed to invoke tool '{tool_name}' with args: {tool_args}, got error: {error_description}"
                raise MCPInvocationError(message, tool_name, tool_args) from e

        message = f"Tool '{tool_name}' failed unexpectedly after all retry attempts"
        raise MCPInvocationError(message, tool_name, tool_args)

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
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
        | tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            Any,
        ],
        connection_type: str,
    ) -> list[types.Tool]:
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

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str | Secret] | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        """
        Initialize a stdio MCP client.

        :param command: Command to run (e.g., "python", "node")
        :param args: Arguments to pass to the command
        :param env: Environment variables for the command
        :param max_retries: Maximum number of reconnection attempts
        :param base_delay: Base delay for exponential backoff in seconds
        """
        super().__init__(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay)
        self.command: str = command
        self.args: list[str] = args or []
        # Resolve Secret values in environment variables
        self.env: dict[str, str] | None = None
        if env:
            self.env = {
                key: (value.resolve_value() if isinstance(value, Secret) else value) or "" for key, value in env.items()
            }
        logger.debug(f"PROCESS: Created StdioClient for command: {command} {' '.join(self.args or [])}")

    async def connect(self) -> list[types.Tool]:
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

    def __init__(
        self, server_info: "SSEServerInfo", max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
    ) -> None:
        """
        Initialize an SSE MCP client using server configuration.

        :param server_info: Configuration object containing URL, token, timeout, etc.
        :param max_retries: Maximum number of reconnection attempts
        :param base_delay: Base delay for exponential backoff in seconds
        """
        super().__init__(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay)

        # in post_init we validate the url and set the url field so it is guaranteed to be valid
        # safely ignore the mypy warning here
        self.url: str = server_info.url  # type: ignore[assignment]
        self.token: str | None = (
            server_info.token.resolve_value() if isinstance(server_info.token, Secret) else server_info.token
        )
        # Resolve Secret values in headers dictionary
        self.headers: dict[str, str] | None = _resolve_headers(server_info.headers)
        self.timeout: int = server_info.timeout

    async def connect(self) -> list[types.Tool]:
        """
        Connect to an MCP server using SSE transport.

        Note: If both custom headers and token are provided, custom headers take precedence.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        if sse_client is None:
            message = (
                "SSE client is not available. "
                "This may require a newer version of the mcp package that includes mcp.client.sse"
            )
            raise MCPConnectionError(message=message, operation="sse_connect")

        # Use custom headers if provided, otherwise fall back to token-based Authorization
        headers = None
        if self.headers:
            headers = self.headers
        elif self.token:
            headers = {"Authorization": f"Bearer {self.token}"}

        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(self.url, headers=headers, timeout=self.timeout)
        )
        return await self._initialize_session_with_transport(sse_transport, f"HTTP server at {self.url}")


class StreamableHttpClient(MCPClient):
    """
    MCP client that connects to servers using streamable HTTP transport.
    """

    def __init__(
        self,
        server_info: "StreamableHttpServerInfo",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        """
        Initialize a streamable HTTP MCP client using server configuration.

        :param server_info: Configuration object containing URL, token, timeout, etc.
        :param max_retries: Maximum number of reconnection attempts
        :param base_delay: Base delay for exponential backoff in seconds
        """
        super().__init__(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay)

        self.url: str = server_info.url
        self.token: str | None = (
            server_info.token.resolve_value() if isinstance(server_info.token, Secret) else server_info.token
        )
        # Resolve Secret values in headers dictionary
        self.headers: dict[str, str] | None = _resolve_headers(server_info.headers)
        self.timeout: int = server_info.timeout

    async def connect(self) -> list[types.Tool]:
        """
        Connect to an MCP server using streamable HTTP transport.

        Note: If both custom headers and token are provided, custom headers take precedence.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        if streamablehttp_client is None:
            message = (
                "Streamable HTTP client is not available. "
                "This may require a newer version of the mcp package that includes mcp.client.streamable_http"
            )
            raise MCPConnectionError(message=message, operation="streamable_http_connect")

        # Use custom headers if provided, otherwise fall back to token-based Authorization
        headers = None
        if self.headers:
            headers = self.headers
        elif self.token:
            headers = {"Authorization": f"Bearer {self.token}"}

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
        result: dict[str, Any] = {"type": generate_qualified_class_name(type(self))}

        # Add all fields from the dataclass
        for dataclass_field in fields(self):
            value = getattr(self, dataclass_field.name)
            if hasattr(value, "to_dict"):
                result[dataclass_field.name] = value.to_dict()
            elif isinstance(value, dict):
                result[dataclass_field.name] = {
                    k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in value.items()
                }
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

        secret_types = {e.value for e in SecretType}
        field_names = {f.name for f in fields(cls)}

        # Iterate over a static list of items to avoid mutation issues
        for name, value in list(data_copy.items()):
            if name not in field_names or not isinstance(value, dict):
                continue

            # Top-level secret?
            if value.get("type") in secret_types:
                deserialize_secrets_inplace(data_copy, keys=[name])
                continue

            # Nested secrets (one level deep)
            nested_keys: list[str] = [
                k for k, v in value.items() if isinstance(v, dict) and v.get("type") in secret_types
            ]
            if nested_keys:
                deserialize_secrets_inplace(value, keys=nested_keys)

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

    For custom headers (e.g., non-standard authentication):

    ```python
    # Single custom header with Secret
    server_info = SSEServerInfo(
        url="https://my-mcp-server.com",
        headers={"X-API-Key": Secret.from_env_var("API_KEY")},
    )

    # Multiple headers (mix of Secret and plain strings)
    server_info = SSEServerInfo(
        url="https://my-mcp-server.com",
        headers={
            "X-API-Key": Secret.from_env_var("API_KEY"),
            "X-Client-ID": "my-client-id",
        },
    )
    ```

    :param url: Full URL of the MCP server (including /sse endpoint)
    :param base_url: Base URL of the MCP server (deprecated, use url instead)
    :param token: Authentication token for the server (optional, generates "Authorization: Bearer `<token>`" header)
    :param headers: Custom HTTP headers (optional, takes precedence over token parameter if provided)
    :param timeout: Connection timeout in seconds
    """

    url: str | None = None
    base_url: str | None = None  # deprecated
    token: str | Secret | None = None
    headers: dict[str, str | Secret] | None = None
    timeout: int = 30
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0

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

        elif self.url and not is_valid_http_url(self.url):
            message = f"Invalid url: {self.url}"
            raise ValueError(message)

    def create_client(self) -> MCPClient:
        """
        Create an SSE MCP client.

        :returns: Configured MCPClient instance
        """
        # Pass the validated SSEServerInfo instance directly
        return SSEClient(
            server_info=self, max_retries=self.max_retries, base_delay=self.base_delay, max_delay=self.max_delay
        )


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

    For custom headers (e.g., non-standard authentication):

    ```python
    # Single custom header with Secret
    server_info = StreamableHttpServerInfo(
        url="https://my-mcp-server.com",
        headers={"X-API-Key": Secret.from_env_var("API_KEY")},
    )

    # Multiple headers (mix of Secret and plain strings)
    server_info = StreamableHttpServerInfo(
        url="https://my-mcp-server.com",
        headers={
            "X-API-Key": Secret.from_env_var("API_KEY"),
            "X-Client-ID": "my-client-id",
        },
    )
    ```

    :param url: Full URL of the MCP server (streamable HTTP endpoint)
    :param token: Authentication token for the server (optional, generates "Authorization: Bearer `<token>`" header)
    :param headers: Custom HTTP headers (optional, takes precedence over token parameter if provided)
    :param timeout: Connection timeout in seconds
    """

    url: str
    token: str | Secret | None = None
    headers: dict[str, str | Secret] | None = None
    timeout: int = 30
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0

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
        return StreamableHttpClient(
            server_info=self, max_retries=self.max_retries, base_delay=self.base_delay, max_delay=self.max_delay
        )


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
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0

    def create_client(self) -> MCPClient:
        """
        Create a stdio MCP client.

        :returns: Configured StdioMCPClient instance
        """
        return StdioClient(
            self.command,
            self.args,
            self.env,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
        )


class MCPTool(Tool):
    """
    A Tool that represents a single tool from an MCP server.

    This implementation uses the official MCP SDK for protocol handling while maintaining
    compatibility with the Haystack tool ecosystem.

    Response handling:
    - Text and image content are supported and returned as JSON strings
    - The JSON contains the structured response from the MCP server
    - Use json.loads() to parse the response into a dictionary

    State-mapping support:
    - MCPTool supports state-mapping parameters (`outputs_to_string`, `inputs_from_state`, `outputs_to_state`)
    - These enable integration with Agent state for automatic parameter injection and output handling
    - See the `__init__` method documentation for details on each parameter

    Example using Streamable HTTP:
    ```python
    import json
    from haystack_integrations.tools.mcp import MCPTool, StreamableHttpServerInfo

    # Create tool instance
    tool = MCPTool(
        name="multiply",
        server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp")
    )

    # Use the tool and parse result
    result_json = tool.invoke(a=5, b=3)
    result = json.loads(result_json)
    ```

    Example using SSE (deprecated):
    ```python
    import json
    from haystack.tools import MCPTool, SSEServerInfo

    # Create tool instance
    tool = MCPTool(
        name="add",
        server_info=SSEServerInfo(url="http://localhost:8000/sse")
    )

    # Use the tool and parse result
    result_json = tool.invoke(a=5, b=3)
    result = json.loads(result_json)
    ```

    Example using stdio:
    ```python
    import json
    from haystack.tools import MCPTool, StdioServerInfo

    # Create tool instance
    tool = MCPTool(
        name="get_current_time",
        server_info=StdioServerInfo(command="python", args=["path/to/server.py"])
    )

    # Use the tool and parse result
    result_json = tool.invoke(timezone="America/New_York")
    result = json.loads(result_json)
    ```
    """

    def __init__(
        self,
        name: str,
        server_info: MCPServerInfo,
        description: str | None = None,
        connection_timeout: int = 30,
        invocation_timeout: int = 30,
        eager_connect: bool = False,
        outputs_to_string: dict[str, Any] | None = None,
        inputs_from_state: dict[str, str] | None = None,
        outputs_to_state: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initialize the MCP tool.

        :param name: Name of the tool to use
        :param server_info: Server connection information
        :param description: Custom description (if None, server description will be used)
        :param connection_timeout: Timeout in seconds for server connection
        :param invocation_timeout: Default timeout in seconds for tool invocations
        :param eager_connect: If True, connect to server during initialization.
                             If False (default), defer connection until warm_up or first tool use,
                             whichever comes first.
        :param outputs_to_string: Optional dictionary defining how tool outputs should be converted into a string.
                                 If the source is provided only the specified output key is sent to the handler.
                                 If the source is omitted the whole tool result is sent to the handler.
                                 Example: `{"source": "docs", "handler": my_custom_function}`
        :param inputs_from_state: Optional dictionary mapping state keys to tool parameter names.
                                 Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
        :param outputs_to_state: Optional dictionary defining how tool outputs map to keys within state as well as
                                optional handlers. If the source is provided only the specified output key is sent
                                to the handler.
                                Example with source: `{"documents": {"source": "docs", "handler": custom_handler}}`
                                Example without source: `{"documents": {"handler": custom_handler}}`
        :raises MCPConnectionError: If connection to the server fails
        :raises MCPToolNotFoundError: If no tools are available or the requested tool is not found
        :raises TimeoutError: If connection times out
        """

        # Store connection parameters for serialization
        self._server_info = server_info
        self._connection_timeout = connection_timeout
        self._invocation_timeout = invocation_timeout
        self._eager_connect = eager_connect
        self._outputs_to_string = outputs_to_string
        self._inputs_from_state = inputs_from_state
        self._outputs_to_state = outputs_to_state
        self._client: MCPClient | None = None
        self._worker: _MCPClientSessionManager | None = None
        self._lock = threading.RLock()

        # don't connect now; initialize permissively
        if not eager_connect:
            # Permissive placeholder JSON Schema so the Tool is valid
            # without discovering the remote schema during validation.
            # Tool parameters/schema will be replaced with the correct schema (from the MCP server) on first use.
            params = {"type": "object", "properties": {}, "additionalProperties": True}
            super().__init__(
                name=name,
                description=description or "",
                parameters=params,
                function=self._invoke_tool,
                outputs_to_string=outputs_to_string,
                inputs_from_state=inputs_from_state,
                outputs_to_state=outputs_to_state,
            )
            return

        logger.debug(f"TOOL: Initializing MCPTool '{name}'")

        try:
            logger.debug(f"TOOL: Connecting to MCP server for '{name}'")
            tool_info = self._connect_and_initialize(name)
            logger.debug(f"TOOL: Found tool '{name}', initializing Tool parent class")

            # Initialize the parent class
            super().__init__(
                name=name,
                description=description or tool_info.description or "",
                parameters=tool_info.inputSchema,
                function=self._invoke_tool,
                outputs_to_string=outputs_to_string,
                inputs_from_state=inputs_from_state,
                outputs_to_state=outputs_to_state,
            )

            # Remove inputs_from_state keys from parameters schema if present
            # This matches the behavior of ComponentTool
            if inputs_from_state and "properties" in self.parameters:
                for key in inputs_from_state.values():
                    self.parameters["properties"].pop(key, None)
                    if "required" in self.parameters and key in self.parameters["required"]:
                        self.parameters["required"].remove(key)

            logger.debug(f"TOOL: Initialization complete for '{name}'")

        except Exception as e:
            # We need to close because we could connect properly, retrieve tools yet
            # fail because of an MCPToolNotFoundError
            self.close()

            # Extract more detailed error information from TaskGroup/ExceptionGroup exceptions
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

    def _connect_and_initialize(self, tool_name: str) -> types.Tool:
        """
        Connect to the MCP server and retrieve the tool schema.

        :param tool_name: Name of the tool to look for
        :returns: The tool schema for this tool
        :raises MCPToolNotFoundError: If the tool is not found on the server
        """
        client = self._server_info.create_client()
        worker = _MCPClientSessionManager(client, timeout=self._connection_timeout)
        tools = worker.tools()

        # Handle no tools case
        if not tools:
            message = "No tools available on server"
            raise MCPToolNotFoundError(message, tool_name=tool_name)

        # Find the specified tool
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool is None:
            available = [t.name for t in tools]
            msg = f"Tool '{tool_name}' not found on server. Available tools: {', '.join(available)}"
            raise MCPToolNotFoundError(msg, tool_name=tool_name, available_tools=available)

        # Publish connection
        self._client = client
        self._worker = worker

        return tool

    def _invoke_tool(self, **kwargs: Any) -> str:
        """
        Synchronous tool invocation.

        :param kwargs: Arguments to pass to the tool
        :returns: JSON string representation of the tool invocation result
        """
        logger.debug(f"TOOL: Invoking tool '{self.name}' with args: {kwargs}")
        try:
            # Connect on first use if eager_connect is turned off
            self.warm_up()

            async def invoke():
                logger.debug(f"TOOL: Inside invoke coroutine for '{self.name}'")
                client = cast(MCPClient, self._client)
                result = await asyncio.wait_for(client.call_tool(self.name, kwargs), timeout=self._invocation_timeout)
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

    async def ainvoke(self, **kwargs: Any) -> str:
        """
        Asynchronous tool invocation.

        :param kwargs: Arguments to pass to the tool
        :returns: JSON string representation of the tool invocation result
        :raises MCPInvocationError: If the tool invocation fails
        :raises TimeoutError: If the operation times out
        """
        try:
            self.warm_up()
            client = cast(MCPClient, self._client)
            return await asyncio.wait_for(client.call_tool(self.name, kwargs), timeout=self._invocation_timeout)
        except asyncio.TimeoutError as e:
            message = f"Tool invocation timed out after {self._invocation_timeout} seconds"
            raise TimeoutError(message) from e
        except Exception as e:
            if isinstance(e, MCPError):
                raise
            message = f"Failed to invoke tool '{self.name}' with args: {kwargs} , got error: {e!s}"
            raise MCPInvocationError(message, self.name, kwargs) from e

    def warm_up(self) -> None:
        """Connect and fetch the tool schema if eager_connect is turned off."""
        with self._lock:
            if self._client is not None:
                return
            tool = self._connect_and_initialize(self.name)
            self.parameters = tool.inputSchema

            # Remove inputs_from_state keys from parameters schema if present
            # This matches the behavior of ComponentTool
            if self._inputs_from_state and "properties" in self.parameters:
                for key in self._inputs_from_state.values():
                    self.parameters["properties"].pop(key, None)
                    if "required" in self.parameters and key in self.parameters["required"]:
                        self.parameters["required"].remove(key)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the MCPTool to a dictionary.

        The serialization preserves all information needed to recreate the tool,
        including server connection parameters, timeout settings, and state-mapping parameters.
        Note that the active connection is not maintained.

        :returns: Dictionary with serialized data in the format:
                  `{"type": fully_qualified_class_name, "data": {parameters}}`
        """
        serialized = {
            "name": self.name,
            "description": self.description,
            "server_info": self._server_info.to_dict(),
            "connection_timeout": self._connection_timeout,
            "invocation_timeout": self._invocation_timeout,
            "eager_connect": self._eager_connect,
            "outputs_to_string": self._outputs_to_string,
            "inputs_from_state": self._inputs_from_state,
            "outputs_to_state": self._outputs_to_state,
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
        including recreating the server_info object and state-mapping parameters.
        A new connection will be established to the MCP server during initialization.

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
        server_info = cast(MCPServerInfo, server_info_class).from_dict(server_info_dict)

        # Handle backward compatibility for timeout parameters
        connection_timeout = inner_data.get("connection_timeout", 30)
        invocation_timeout = inner_data.get("invocation_timeout", 30)
        eager_connect = inner_data.get("eager_connect", False)  # because False is the default

        # Handle state-mapping parameters
        outputs_to_string = inner_data.get("outputs_to_string")
        inputs_from_state = inner_data.get("inputs_from_state")
        outputs_to_state = inner_data.get("outputs_to_state")

        # Create a new MCPTool instance with the deserialized parameters
        # This will establish a new connection to the MCP server
        return cls(
            name=inner_data["name"],
            description=inner_data.get("description"),
            server_info=server_info,
            connection_timeout=connection_timeout,
            invocation_timeout=invocation_timeout,
            eager_connect=eager_connect,
            outputs_to_string=outputs_to_string,
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
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
        self._tools_promise: Future[list[types.Tool]] = Future()

        # Kick off the worker coroutine in the background loop
        self._worker_future, self._stop_event = self.executor.run_background(self._run, timeout=None)

        # Wait (in the caller thread) until connect() finishes or raises.
        try:
            self._tools_promise.result(timeout)
        except BaseException:
            # If connect failed we should cancel the worker so it doesn't hang.
            self.stop()
            raise

    def tools(self) -> list[types.Tool]:
        """Return the tool list already collected during startup."""

        return self._tools_promise.result()

    def stop(self) -> None:
        """Request the worker to shut down and block until done."""

        def _set(ev: asyncio.Event) -> None:
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

    async def _run(self, stop_event: asyncio.Event) -> None:
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
