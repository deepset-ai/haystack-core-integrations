# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Full example of a pipeline that uses MCPTool to get the current time
# and then uses the time to answer a user question.
# Here we use the mcp-server-time mcp package
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time

import logging

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp.mcp_tool import MCPTool, StdioServerInfo

# Setup targeted logging - only show debug logs from our package
logging.basicConfig(level=logging.WARNING)  # Set root logger to WARNING
mcp_logger = logging.getLogger("haystack_integrations.tools.mcp")
mcp_logger.setLevel(logging.DEBUG)
# Ensure we have at least one handler to avoid messages going to root logger
if not mcp_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    mcp_logger.addHandler(handler)
    mcp_logger.propagate = False  # Prevent propagation to root logger


def main():
    time_tool = None
    try:
        time_tool = MCPTool(
            name="get_current_time",
            server_info=StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"]),
        )
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[time_tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[time_tool]))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        user_input = "What is the time in New York? Be brief."  # can be any city
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        print(result["response_llm"]["replies"][0].text)
    finally:
        if time_tool:
            time_tool.close()


if __name__ == "__main__":
    main()
