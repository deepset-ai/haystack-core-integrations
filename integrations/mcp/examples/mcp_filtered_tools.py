# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo

# This example demonstrates using MCPToolset with SSE transport
# and filtering tools by name
# Run this client after running the server mcp_server.py with sse transport
# It shows how MCPToolset can selectively include only specific tools


def main():
    """Example of using MCPToolset with filtered tools."""

    full_toolset = None
    filtered_toolset = None
    try:
        print("Creating toolset with all available tools:")
        # Create a toolset with all available tools
        full_toolset = MCPToolset(
            server_info=SSEServerInfo(url="http://localhost:8000/sse"),
        )

        # Print all discovered tools
        print(f"Discovered {len(full_toolset)} tools:")
        for tool in full_toolset:
            print(f"  - {tool.name}: {tool.description}")

        print("\nCreating toolset with filtered tools:")
        # Create a toolset with only specific tools
        # In this example, we're only including the 'add' tool
        filtered_toolset = MCPToolset(
            server_info=SSEServerInfo(url="http://localhost:8000/sse"),
            tool_names=["add"],  # Only include the 'add' tool
        )

        # Print filtered tools
        print(f"Filtered toolset has {len(filtered_toolset)} tools:")
        for tool in filtered_toolset:
            print(f"  - {tool.name}: {tool.description}")

        # Use the filtered toolset
        if len(filtered_toolset) > 0:
            add_tool = filtered_toolset.tools[0]  # The only tool should be 'add'
            result = add_tool.invoke(a=10, b=5)
            print(f"\nInvoking {add_tool.name}: 10 + 5 = {result.content[0].text}")
        else:
            print("No tools available in the filtered toolset")

    except Exception as e:
        print(f"Error in filtered toolset example: {e}")
    finally:
        if full_toolset:
            full_toolset.close()
        if filtered_toolset:
            filtered_toolset.close()


if __name__ == "__main__":
    main()
