# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo

# run this client after running the server mcp_sse_server.py
# it shows how easy it is to use the MCPTool with SSE transport


def main():
    """Example of synchronous usage of MCPTool with SSE transport."""

    server_info = SSEServerInfo(
        base_url="http://localhost:8000",
    )

    tool = MCPTool(name="add", server_info=server_info)

    tool_subtract = MCPTool(name="subtract", server_info=server_info)

    try:
        print(f"Tool spec: {tool.tool_spec}")

        result = tool.invoke(a=7, b=3)
        print(f"7 + 3 = {result}")

        result = tool_subtract.invoke(a=5, b=3)
        print(f"5 - 3 = {result}")

        result = tool.invoke(a=10, b=20)
        print(f"10 + 20 = {result}")

    except Exception as e:
        print(f"Error in sync example: {e}")
    finally:
        pass


if __name__ == "__main__":
    main()
