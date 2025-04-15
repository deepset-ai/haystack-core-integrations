# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example of a Hayhooks PipelineWrapper for deploying an MCP Tool-based time pipeline as a REST API.

To run this example:

1. Install Hayhooks and dependencies:
   $ pip install hayhooks haystack-ai

2. Start the Hayhooks server:
   $ hayhooks run

3. Deploy this pipeline wrapper:
   $ hayhooks pipeline deploy-files -n time_pipeline {root_dir_for_mcp_haystack_integration}/examples/hayhooks/

4. Invoke via curl:
   $ curl -X POST 'http://localhost:1416/time_pipeline/run' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"query":"What is the time in San Francisco? Be brief"}'

For more information, see: https://github.com/deepset-ai/hayhooks
"""

from hayhooks import BasePipelineWrapper
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp.mcp_tool import MCPTool, StdioServerInfo


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """
        Setup the pipeline with MCP time tool.

        This creates a pipeline that uses an MCP time tool to get the current time
        and then uses the time to answer a user question.
        """

        time_tool = MCPTool(
            name="get_current_time",
            server_info=StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"]),
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[time_tool]))
        self.pipeline.add_component("tool_invoker", ToolInvoker(tools=[time_tool]))
        self.pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        self.pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))
        self.pipeline.connect("llm.replies", "tool_invoker.messages")
        self.pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        self.pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        self.pipeline.connect("adapter.output", "response_llm.messages")

    def run_api(self, query: str) -> str:
        """
        Run the pipeline with a user query.

        :param query: The user query asking about time
        :return: The response from the LLM
        """
        # Create a user message from the query
        user_input_msg = ChatMessage.from_user(text=query)

        # Run the pipeline
        result = self.pipeline.run(
            {"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}}
        )

        # Return the text of the first reply
        return result["response_llm"]["replies"][0].text
