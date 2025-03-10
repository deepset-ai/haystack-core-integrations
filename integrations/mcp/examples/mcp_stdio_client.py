# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa: T201
from haystack_integrations.tools.mcp import MCPTool, StdioMCPServerInfo

# For stdio MCPTool we don't need to run a server, we can just use the MCPTool directly
# Here we use the mcp-server-time server
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time


def main():
    """Example of using the MCPTool implementation with stdio transport."""

    try:

        stdio_tool = MCPTool(
            name="get_current_time",
            server_info=StdioMCPServerInfo(
                command="python", args=["-m", "mcp_server_time", "--local-timezone=Europe/Berlin"]
            ),
        )

        print(f"Tool spec: {stdio_tool.tool_spec}")

        result = stdio_tool.invoke(timezone="America/New_York")
        print(f"Current time in New York: {result}")

        result = stdio_tool.invoke(timezone="America/San_Francisco")
        print(f"Current time in San Francisco: {result}")
    except Exception as e:
        print(f"Error in stdio example: {e}")


if __name__ == "__main__":
    main()
