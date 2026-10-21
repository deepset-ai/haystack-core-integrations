# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.utils import Secret

from haystack_integrations.tools.mcp import MCPToolset, OAuthConfig, StreamableHttpServerInfo

# This example demonstrates connecting to an MCP server protected with OAuth 2.0, using the Client Credentials grant
# Run this client after running the server mcp_oauth_server.py, with the client secret in an environment variable:
#
#   export MCP_CLIENT_SECRET=haystack-secret
#   python mcp_oauth_client.py
#
# The toolset discovers the authorization server from the MCP server, requests an access token with the client
# credentials, and sends it with every request. When the token expires, a new one is requested automatically.


def main():
    """Example of using MCPToolset with an OAuth-protected MCP server."""

    server_info = StreamableHttpServerInfo(
        url="http://127.0.0.1:8000/mcp",
        oauth_config=OAuthConfig(
            client_id="haystack-client",
            client_secret=Secret.from_env_var("MCP_CLIENT_SECRET"),
            # Only send the client credentials to this authorization server (needs mcp>=1.30.0 or mcp>=2.2.0)
            issuer="http://127.0.0.1:9000",
        ),
    )

    toolset = None
    try:
        toolset = MCPToolset(server_info=server_info, eager_connect=True)

        print(f"Discovered {len(toolset)} tools:")
        for tool in toolset:
            print(f"  - {tool.name}: {tool.description}")

        add_tool = next(tool for tool in toolset if tool.name == "add")
        result = add_tool.invoke(a=7, b=3)
        print(f"\n7 + 3 = {result}")

        # The serialized toolset contains a reference to the environment variable, never the secret or the token
        print(f"\nSerialized OAuth config: {toolset.to_dict()['data']['server_info']['oauth_config']}")

    except Exception as e:
        print(f"Error in OAuth client example: {e}")
    finally:
        if toolset:
            toolset.close()


if __name__ == "__main__":
    main()
