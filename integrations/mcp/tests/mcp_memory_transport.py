from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import types
from mcp.server import Server

from haystack_integrations.tools.mcp import MCPClient, MCPInvocationError, MCPServerInfo
from haystack_integrations.tools.mcp.compatibility_layer import MCP_V2, mcp_field_value

if MCP_V2:
    from mcp import Client

    @asynccontextmanager
    async def _connected_session(server):
        async with Client(server) as client:
            yield client

else:
    from mcp.shared.memory import create_connected_server_and_client_session

    @asynccontextmanager
    async def _connected_session(server):
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            yield session


class InMemoryClient(MCPClient):
    """
    MCP client that connects to servers using in-memory transport.
    """

    def __init__(self, server: Server) -> None:
        super().__init__()
        self.server: Server = server

    async def connect(self) -> list[types.Tool]:
        """
        Connect to an MCP server using stdio transport.


        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        async with _connected_session(self.server) as session:
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
        """
        async with _connected_session(self.server) as session:
            response = await session.call_tool(tool_name, tool_args)
            if mcp_field_value(model=response, name="is_error"):
                raise MCPInvocationError(
                    message=f"Tool '{tool_name}' returned an error: {response.content!s}",
                    tool_name=tool_name,
                )
            return response.model_dump_json(by_alias=True)


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
