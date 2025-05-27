# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo, StreamableHttpServerInfo

# Setup targeted logging - only show debug logs from our package
logging.basicConfig(level=logging.WARNING)  # Set root logger to WARNING
mcp_logger = logging.getLogger("haystack_integrations.tools.mcp")
mcp_logger.setLevel(logging.DEBUG)
# Ensure we have at least one handler to avoid messages going to root logger
if not mcp_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    mcp_logger.addHandler(handler)
    mcp_logger.propagate = False  # Prevent propagation to root logger

# Run this client after running the server mcp_server.py
# It shows how easy it is to use the MCPTool with different transport options


def main():
    """Example of MCPTool usage with server connection."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an MCP client to connect to the server")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport mechanism for the MCP client (default: sse)",
    )
    args = parser.parse_args()

    # Construct the appropriate URL based on transport type
    if args.transport == "sse":
        server_info = SSEServerInfo(url="http://localhost:8000/sse")
    else:  # streamable-http
        server_info = StreamableHttpServerInfo(url="http://localhost:8000/mcp")
    tool = None
    tool_subtract = None
    try:
        tool = MCPTool(name="add", server_info=server_info)
        tool_subtract = MCPTool(name="subtract", server_info=server_info)

        result = tool.invoke(a=7, b=3)
        print(f"7 + 3 = {result}")

        result = tool_subtract.invoke(a=5, b=3)
        print(f"5 - 3 = {result}")

        result = tool.invoke(a=10, b=20)
        print(f"10 + 20 = {result}")

    except Exception as e:
        print(f"Error in client example: {e}")
    finally:
        if tool:
            tool.close()
        if tool_subtract:
            tool_subtract.close()


if __name__ == "__main__":
    main()
