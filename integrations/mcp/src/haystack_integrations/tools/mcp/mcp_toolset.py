import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool, Toolset

# Import MCP-related classes
from .mcp_tool import (
    MCPTool,
    SSEClient,
    SSEServerInfo,
)

logger = logging.getLogger(__name__)


class MCPServerError(Exception):
    """Base class for MCP server-related errors."""

    def __init__(self, message: str) -> None:
        """
        Initialize the MCPServerError.

        :param message: Descriptive error message
        """
        super().__init__(message)
        self.message = message


class MCPServerConnectionError(MCPServerError):
    """Error connecting to MCP server."""

    def __init__(self, message: str, server_info: Optional[SSEServerInfo] = None) -> None:
        """
        Initialize the MCPServerConnectionError.

        :param message: Descriptive error message
        :param server_info: Server connection information that was used
        """
        super().__init__(message)
        self.server_info = server_info


@dataclass
class MCPServer(Toolset):
    """
    A Toolset that connects to an MCP server and provides access to its tools.

    This implementation connects to a remote MCP server using SSE (Server-Sent Events)
    and dynamically loads the available tools from the server.

    Example:
    ```python
    from haystack.tools import MCPServer, SSEServerInfo
    from haystack.components.tools import ToolInvoker

    # Create server info
    server_info = SSEServerInfo(
        base_url="http://localhost:8000",
        token="your-auth-token",  # Optional
        timeout=30  # Optional, defaults to 30 seconds
    )

    # Create the MCP server toolset
    mcp_server = MCPServer(server_info=server_info)

    # Use the toolset with a ToolInvoker or ChatGenerator component
    invoker = ToolInvoker(tools=mcp_server)
    ```

    The MCPServer class handles:
    - Connecting to the MCP server
    - Discovering available tools
    - Creating Tool instances for each available tool
    - Managing the connection lifecycle
    - Serialization and deserialization of the server configuration
    """

    server_info: SSEServerInfo
    connection_timeout: int = 30
    invocation_timeout: int = 30
    tools: List[Tool] = field(default_factory=list, init=False)

    def __post_init__(self):
        """
        Initialize the MCP server and load tools.

        This method connects to the MCP server and loads all available tools.
        """
        try:
            # Create the SSE client
            client = SSEClient(
                self.server_info.base_url,
                self.server_info.token,
                self.server_info.timeout
            )

            # Connect and get available tools
            tools = self._run_sync(client.connect(), timeout=self.connection_timeout)

            # Create MCPTool instances for each available tool
            for tool_info in tools:
                tool = MCPTool(
                    name=tool_info.name,
                    server_info=self.server_info,
                    description=tool_info.description,
                    connection_timeout=self.connection_timeout,
                    invocation_timeout=self.invocation_timeout
                )
                self.tools.append(tool)

            # Check for duplicate tool names
            self._check_duplicate_tool_names()

        except Exception as e:
            message = f"Failed to initialize MCPServer: {e}"
            raise MCPServerConnectionError(message, self.server_info) from e

    def _check_duplicate_tool_names(self):
        """
        Check for duplicate tool names in the toolset.

        :raises ValueError: If duplicate tool names are found
        """
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            raise ValueError(f"Duplicate tool names found: {duplicates}")

    def _run_sync(self, coro, timeout: Optional[float] = None):
        """
        Run an async coroutine synchronously.

        :param coro: The coroutine to run
        :param timeout: Optional timeout in seconds
        :returns: The result of the coroutine
        :raises TimeoutError: If the operation times out
        """
        try:
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Operation timed out after {timeout} seconds") from e

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the MCPServer to a dictionary.

        :returns: A dictionary representation of the MCPServer
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
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServer":
        """
        Deserialize an MCPServer from a dictionary.

        :param data: Dictionary representation of the MCPServer
        :returns: A new MCPServer instance
        """
        inner_data = data["data"]
        
        # Reconstruct the server_info object
        server_info_dict = inner_data.get("server_info", {})
        server_info_class = import_class_by_name(server_info_dict["type"])
        server_info = server_info_class.from_dict(server_info_dict)
        
        # Create a new MCPServer instance
        return cls(
            server_info=server_info,
            connection_timeout=inner_data.get("connection_timeout", 30),
            invocation_timeout=inner_data.get("invocation_timeout", 30),
        )

    def __iter__(self):
        """
        Return an iterator over the Tools in this MCPServer.

        :returns: An iterator yielding Tool instances
        """
        return iter(self.tools)

    def __contains__(self, item: Any) -> bool:
        """
        Check if a tool is in this MCPServer.

        Supports checking by:
        - Tool instance: tool in mcp_server
        - Tool name: "tool_name" in mcp_server

        :param item: Tool instance or tool name string
        :returns: True if contained, False otherwise
        """
        if isinstance(item, str):
            return any(tool.name == item for tool in self.tools)
        if isinstance(item, Tool):
            return item in self.tools
        return False

    def __len__(self) -> int:
        """
        Return the number of Tools in this MCPServer.

        :returns: Number of Tools
        """
        return len(self.tools)

    def __getitem__(self, index):
        """
        Get a Tool by index.

        :param index: Index of the Tool to get
        :returns: The Tool at the specified index
        """
        return self.tools[index]

    def __add__(self, other: Union[Tool, "MCPServer", List[Tool]]) -> "MCPServer":
        """
        Concatenate this MCPServer with another Tool, MCPServer, or list of Tools.

        :param other: Another Tool, MCPServer, or list of Tools to concatenate
        :returns: A new MCPServer containing all tools
        :raises TypeError: If the other parameter is not a Tool, MCPServer, or list of Tools
        :raises ValueError: If the combination would result in duplicate tool names
        """
        if isinstance(other, Tool):
            combined_tools = self.tools + [other]
        elif isinstance(other, (MCPServer, Toolset)):
            combined_tools = self.tools + list(other)
        elif isinstance(other, list) and all(isinstance(item, Tool) for item in other):
            combined_tools = self.tools + other
        else:
            raise TypeError(f"Cannot add {type(other).__name__} to MCPServer")

        # Check for duplicates
        tool_names = [tool.name for tool in combined_tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            raise ValueError(f"Duplicate tool names found: {duplicates}")

        # Create a new MCPServer with the combined tools
        new_server = MCPServer(
            server_info=self.server_info,
            connection_timeout=self.connection_timeout,
            invocation_timeout=self.invocation_timeout,
        )
        new_server.tools = combined_tools
        return new_server