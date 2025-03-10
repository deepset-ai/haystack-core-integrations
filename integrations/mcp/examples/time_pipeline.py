# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Full example of a pipeline that uses MCPTool to get the current time
# and then uses the time to answer a user question.
# Here we use the mcp-server-time mcp package
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp.mcp_tool import MCPTool, StdioMCPServerInfo


def main():
    time_tool = MCPTool(
        name="get_current_time",
        server_info=StdioMCPServerInfo(
            command="python", args=["-m", "mcp_server_time", "--local-timezone=Europe/Berlin"]
        ),
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


if __name__ == "__main__":
    main()
