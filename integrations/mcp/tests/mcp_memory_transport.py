from dataclasses import dataclass
from typing import Any

from haystack.tools import Tool
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session

from haystack_integrations.tools.mcp import MCPClient, MCPServerInfo


class InMemoryClient(MCPClient):
    """
    MCP client that connects to servers using in-memory transport.
    """

    def __init__(self, server: Server) -> None:
        super().__init__()
        self.server: Server = server

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using stdio transport.


        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        async with create_connected_server_and_client_session(self.server) as session:
            await session.initialize()
            response = await session.list_tools()
            return response.tools

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
        async with create_connected_server_and_client_session(self.server) as session:
            await session.initialize()
            response = await session.call_tool(tool_name, tool_args)
            return self._validate_response(tool_name, response)


@dataclass
class InMemoryServerInfo(MCPServerInfo):
    """
    Data class that encapsulates in-memory MCP server connection parameters.

    :param server: MCP server to connect to
    """

    server: Server

    def create_client(self) -> MCPClient:
        """
        Create an in-memory MCP client.

        :returns: Configured InMemoryClient instance
        """
        return InMemoryClient(self.server)
