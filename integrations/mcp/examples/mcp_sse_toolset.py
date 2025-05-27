# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo

# This example demonstrates using MCPToolset with SSE transport
# Run this client after running the server mcp_server.py with sse transport
# It shows how MCPToolset automatically discovers and creates tools from the MCP server


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
    """Example of using MCPToolset with SSE transport."""

    sse_toolset = None
    try:
        # Create the toolset - this automatically discovers all available tools
        # from the MCP server and creates Tool instances for each one
        sse_toolset = MCPToolset(
            server_info=SSEServerInfo(url="http://localhost:8000/sse"),
        )

        # Print discovered tools
        print(f"Discovered {len(sse_toolset)} tools:")
        for tool in sse_toolset:
            print(f"  - {tool.name}: {tool.description}")

        # Get tools by name from the toolset
        add_tool = find_tool(sse_toolset, "add")
        if not add_tool:
            print("Add tool not found!")
            return

        subtract_tool = find_tool(sse_toolset, "subtract")
        if not subtract_tool:
            print("Subtract tool not found!")
            return

        # Use the tools
        result = add_tool.invoke(a=7, b=3)
        print(f"7 + 3 = {result.content[0].text}")

        result = subtract_tool.invoke(a=5, b=3)
        print(f"5 - 3 = {result.content[0].text}")

        result = add_tool.invoke(a=10, b=20)
        print(f"10 + 20 = {result.content[0].text}")

    except Exception as e:
        print(f"Error in SSE toolset example: {e}")

    finally:
        if sse_toolset:
            sse_toolset.close()


if __name__ == "__main__":
    main()
