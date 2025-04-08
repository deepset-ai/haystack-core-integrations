import asyncio
import logging
from typing import Any

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Toolset

# Import MCP-related classes
from .mcp_tool import (
    MCPConnectionError,
    MCPServerInfo,
    MCPTool,
)

logger = logging.getLogger(__name__)


class MCPToolset(Toolset):
    """
    A Toolset that connects to an MCP (Model Context Protocol) server and provides
    access to its tools.

    MCPToolset dynamically discovers and loads tools from any MCP-compliant server,
    supporting both network-based SSE connections and local process-based stdio connections.
    This dual connectivity allows for integrating with both remote and local MCP servers.

    Example using SSE (for remote API services):
    ```python
    from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo
    from haystack.components.tools import ToolInvoker

    # Connect to a remote MCP server via SSE
    server_info = SSEServerInfo(
        base_url="https://api.example.com/mcp",  # Remote server URL
        token="your-auth-token",                 # Authentication token
        timeout=30                               # Connection timeout in seconds
    )

    # Create the toolset with the remote server connection info
    toolset = MCPToolset(server_info=server_info)

    # Tools are automatically discovered and can be used with Haystack components
    invoker = ToolInvoker(tools=toolset)
    result = invoker.run(tool_name="calculator", parameters={"expression": "2+2"})
    ```

    Example using stdio (for local tool processes):
    ```python
    from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo
    from haystack.components.tools import ToolInvoker

    # Connect to a local MCP-compatible tool via stdio
    server_info = StdioServerInfo(
        command="python",                         # Command to execute
        args=["path/to/mcp_tool_server.py"],      # Arguments for the command
        env={"DEBUG": "true"}                     # Optional environment variables
    )

    # Create the toolset - discovery works the same way regardless of connection type
    toolset = MCPToolset(server_info=server_info)

    # Use the toolset in a tool invoker
    invoker = ToolInvoker(tools=toolset)
    result = invoker.run(tool_name="time_tool", parameters={"timezone": "UTC"})
    ```

    Combining with other tools:
    ```python
    from haystack.tools import Tool
    from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo
    from haystack.components.tools import ToolInvoker

    # Create a regular Tool
    def my_python_function(x: int, y: int) -> int:
        return x * y

    multiply_tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        function=my_python_function,
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"}
            },
            "required": ["x", "y"]
        }
    )

    # Create an MCP toolset connected to a remote server
    mcp_tools = MCPToolset(
        server_info=SSEServerInfo(base_url="https://api.example.com/mcp")
    )

    # Combine them (creates a new toolset)
    combined_tools = mcp_tools + [multiply_tool]

    # Use the combined toolset
    invoker = ToolInvoker(tools=combined_tools)
    ```

    The MCPToolset handles connection and cleanup automatically, making it simple
    to integrate MCP-compatible tools into Haystack pipelines.
    """

    def __init__(self, server_info: MCPServerInfo, connection_timeout: int = 30, invocation_timeout: int = 30):
        """
        Initialize the MCP toolset.

        :param server_info: Connection information for the MCP server
        :param connection_timeout: Timeout in seconds for server connection
        :param invocation_timeout: Default timeout in seconds for tool invocations
        """
        # Initialize with empty tools list first
        super().__init__(tools=[])

        # Store configuration
        self.server_info = server_info
        self.connection_timeout = connection_timeout
        self.invocation_timeout = invocation_timeout

        # Connect and load tools
        try:
            # Create the appropriate client using the factory method
            client = self.server_info.create_client()

            # Connect and get available tools
            tools = self._run_sync(client.connect(), timeout=self.connection_timeout)

            # Create MCPTool instances for each available tool and add them
            for tool_info in tools:
                tool = MCPTool(
                    name=tool_info.name,
                    server_info=self.server_info,
                    description=tool_info.description,
                    connection_timeout=self.connection_timeout,
                    invocation_timeout=self.invocation_timeout,
                )
                # Handles duplicates and other validation
                self.add(tool)

        except Exception as e:
            message = f"Failed to initialize MCPToolset: {e}"
            raise MCPConnectionError(message=message, server_info=self.server_info, operation="initialize") from e

    def _run_sync(self, coro, timeout: float | None = None):
        """
        Run an async coroutine synchronously.

        :param coro: The coroutine to run
        :param timeout: Optional timeout in seconds
        :returns: The result of the coroutine
        :raises TimeoutError: If the operation times out
        """
        try:
            # Apply timeout if specified
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)

            # Try to get a running loop first
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If we can't get a running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)

            # If the loop is already running, run our coroutine inline
            if loop.is_running():
                # Create a new loop for our coroutine to avoid conflicts
                # This should prevent "This event loop is already running" errors
                temp_loop = asyncio.new_event_loop()
                try:
                    return temp_loop.run_until_complete(coro)
                finally:
                    temp_loop.close()
            else:
                # Use the existing loop
                return loop.run_until_complete(coro)

        except asyncio.TimeoutError as e:
            message = f"Operation timed out after {timeout} seconds"
            raise TimeoutError(message) from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the MCPToolset to a dictionary.

        :returns: A dictionary representation of the MCPToolset
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "server_info": self.server_info.to_dict(),
                "connection_timeout": self.connection_timeout,
                "invocation_timeout": self.invocation_timeout,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPToolset":
        """
        Deserialize an MCPToolset from a dictionary.

        :param data: Dictionary representation of the MCPToolset
        :returns: A new MCPToolset instance
        """
        inner_data = data["data"]

        # Reconstruct the server_info object
        server_info_dict = inner_data.get("server_info", {})
        server_info_class = import_class_by_name(server_info_dict["type"])
        server_info = server_info_class.from_dict(server_info_dict)

        # Create a new MCPToolset instance
        return cls(
            server_info=server_info,
            connection_timeout=inner_data.get("connection_timeout", 30),
            invocation_timeout=inner_data.get("invocation_timeout", 30),
        )
