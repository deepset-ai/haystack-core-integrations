# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

# This example demonstrates using MCPToolset with streamable-http transport
# and filtering tools by name
# Run this client after running the server mcp_server.py
# It shows how MCPToolset can selectively include only specific tools


def main():
    """Example of using MCPToolset with filtered tools."""

    full_toolset = None
    filtered_toolset = None
    try:
        print("Creating toolset with all available tools:")
        # Create a toolset with all available tools
        full_toolset = MCPToolset(
            server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp"),
            eager_connect=True,
        )

        # Print all discovered tools
        print(f"Discovered {len(full_toolset)} tools:")
        for tool in full_toolset:
            print(f"  - {tool.name}: {tool.description}")

        print("\nCreating toolset with filtered tools:")
        # Create a toolset with only specific tools
        # In this example, we're only including the 'add' tool
        filtered_toolset = MCPToolset(
            server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp"),
            tool_names=["add"],  # Only include the 'add' tool
            eager_connect=True,
        )

        # Print filtered tools
        print(f"Filtered toolset has {len(filtered_toolset)} tools:")
        for tool in filtered_toolset:
            print(f"  - {tool.name}: {tool.description}")

        # Use the filtered toolset
        if len(filtered_toolset) > 0:
            add_tool = filtered_toolset.tools[0]  # The only tool should be 'add'
            result = add_tool.invoke(a=10, b=5)
            print(f"\nInvoking {add_tool.name}: 10 + 5 = {result}")
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
