# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from mcp.server.fastmcp import FastMCP

# run this server first before running the client mcp_filtered_tools.py or mcp_client.py
# it shows how easy it is to create a MCP server in just a few lines of code
# then we'll use the MCPTool to invoke the server


mcp = FastMCP("MCP Calculator")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
