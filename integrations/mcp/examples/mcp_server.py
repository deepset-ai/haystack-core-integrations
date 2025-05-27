# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run an MCP server with different transport options (sse or streamable-http)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport mechanism for the MCP server (default: sse)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)
