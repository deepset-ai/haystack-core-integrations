# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Full example of a pipeline that uses MCPToolset to get the current time
# and then uses the time to answer a user question.
# Here we use the mcp-server-time mcp package
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time

import os

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo


def main():
    # Create server info for the time service
    server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

    mcp_toolset = None
    try:
        # Create the toolset - this will automatically discover all available tools
        mcp_toolset = MCPToolset(server_info)
        # Check if OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY environment variable is not set.")
            print("You need to set it to run the pipeline example.")
            print("For now, demonstrating direct tool usage:")

        pipeline = Pipeline()
        pipeline.add_component(
            "agent", Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"), tools=mcp_toolset)
        )

        user_input = "What is the time in New York? Be brief."  # can be any city
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"agent": {"messages": [user_input_msg]}})

        print(result["agent"]["messages"][-1].text)

    finally:
        if mcp_toolset:
            mcp_toolset.close()


if __name__ == "__main__":
    main()
