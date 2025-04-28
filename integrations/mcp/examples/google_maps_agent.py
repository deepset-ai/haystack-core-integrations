# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

### Overview
# This script shows how to use the **Google Maps MCP Server** in combination with the **Haystack Agent** and **OpenAI's Chat Generator** to search for places via the Google Places API.

# ---

# ### 🔧 Step 1: Start the Google Maps MCP Server

# Make sure you have a valid **Google Maps API key**.
# See [Google Maps API Key](https://developers.google.com/maps/documentation/places/web-service/get-api-key) for more information.

# Run the following Docker command to start the MCP Server locally on port 8000:

# ```bash
# docker run -it --rm -p 8000:8000 \
#   -e GOOGLE_MAPS_API_KEY=$GOOGLE_MAPS_API_KEY \
#   supercorp/supergateway \
#   --stdio "npx -y @modelcontextprotocol/server-google-maps" \
#   --port 8000
# ```

# ---

# ### 🕵️ Step 2: Inspect Available Tools (Optional but useful)

# You can verify that the MCP server is running and see the available tools using:

# ```bash
# npx -y @modelcontextprotocol/inspector
# ```

# Connect MCP Inspector to the server at `http://localhost:8000/sse` and click on "List Tools" to display tools such as:

# - `maps_geocode`: Convert address → coordinates
# - `maps_reverse_geocode`: Convert coordinates → address
# - `maps_search_places`: Search places (e.g., restaurants)
# - `maps_place_details`: Get details of a specific place
# - `maps_distance_matrix`: Calculate travel distance/time
# - `maps_elevation`: Get elevation info
# - `maps_directions`: Get route directions

# ---

# ### ▶️ Step 3: Run the Python Script

# This script sets up:
# - A **Langfuse trace**
# - A **MCPToolset** with selected tools (`maps_geocode`, `maps_search_places`)
# - An **Agent** using `gpt-4.1` to process a natural language request

# To run the script:

# ```bash
# python examples/google_maps_agent.py
# ```

# The agent will respond to this example query:

# > "Find the five best Persian restaurants close to Zinnowitzer Str. 1, 10115 Berlin, Germany"

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

# from haystack_integrations.components.connectors.langfuse.langfuse_connector import LangfuseConnector
from haystack_integrations.tools.mcp.mcp_tool import SSEServerInfo
from haystack_integrations.tools.mcp.mcp_toolset import MCPToolset


def main():
    # tracer = LangfuseConnector("Agent google maps search")
    # Optionally, you can use Langfuse to trace the agent's activity but it needs
    # additional configuration.
    # See [Langfuse integration](https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/langfuse) for more information.
    toolset = MCPToolset(
        SSEServerInfo(base_url="http://localhost:8000"), tool_names=["maps_geocode", "maps_search_places"]
    )
    agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1"), tools=toolset)
    result = agent.run(
        messages=[
            ChatMessage.from_user(
                text="Find the five best persian restaurants close to Zinnowitzer Str. 1, 10115 Berlin, Germany"
            )
        ]
    )
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
