# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

# This example shows how to use MCPToolset with stdio transport
# MCPToolset automatically discovers all available tools from the MCP server
# Here we use the mcp-server-time server
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time


def find_tool(toolset, tool_name):
    """
    Find a tool by name in the toolset.

    :param toolset: The toolset to search in
    :param tool_name: The name of the tool to find
    :returns: The tool if found, None otherwise
    """
    for tool in toolset:
        if tool.name == tool_name:
            return tool
    return None


def main():
    """Example of using the MCPToolset implementation with stdio transport."""

    stdio_toolset = None
    try:
        # Create server info for the time service
        server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

        # Create the toolset - this will automatically discover all available tools
        stdio_toolset = MCPToolset(server_info=server_info)

        # Print discovered tools
        print(f"Discovered {len(stdio_toolset)} tools:")
        for tool in stdio_toolset:
            print(f"  - {tool.name}: {tool.description}")

        # Find tools by name using the helper function
        time_tool = find_tool(stdio_toolset, "get_current_time")
        if not time_tool:
            print("Time tool not found!")
            return

        # Use the get_current_time tool
        result = time_tool.invoke(timezone="America/New_York")
        print(f"Current time in New York: {result.content[0].text}")

        result = time_tool.invoke(timezone="America/Los_Angeles")
        print(f"Current time in Los Angeles: {result.content[0].text}")
    except Exception as e:
        print(f"Error in stdio toolset example: {e}")
    finally:
        if stdio_toolset:
            stdio_toolset.close()


if __name__ == "__main__":
    main()
