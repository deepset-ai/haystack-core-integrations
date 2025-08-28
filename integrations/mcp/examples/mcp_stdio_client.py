# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("haystack_integrations.tools.mcp")
logger.setLevel(logging.DEBUG)

# For stdio MCPTool we don't need to run a server, we can just use the MCPTool directly
# Here we use the mcp-server-time server
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time


def main():
    """Example of using the MCPTool implementation with stdio transport."""

    stdio_tool = None
    try:
        stdio_tool = MCPTool(
            name="get_current_time",
            server_info=StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"]),
        )

        print(f"Tool spec: {stdio_tool.tool_spec}")

        result = stdio_tool.invoke(timezone="America/New_York")
        print(f"Current time in New York: {result}")

        result = stdio_tool.invoke(timezone="America/Los_Angeles")
        print(f"Current time in Los Angeles: {result}")
    except Exception as e:
        print(f"Error in stdio example: {e}")
    finally:
        if stdio_tool:
            stdio_tool.close()


if __name__ == "__main__":
    main()
