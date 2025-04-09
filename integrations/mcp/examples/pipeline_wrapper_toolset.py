# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example of a Hayhooks PipelineWrapper for deploying an MCP Toolset-based time pipeline as a REST API.

This example uses MCPToolset instead of individual MCPTool instances, demonstrating how to 
filter which tools to use from an MCP server.

To run this example:

1. Install Hayhooks and dependencies:
   $ pip install hayhooks haystack-ai

2. Start the Hayhooks server:
   $ hayhooks run

3. Deploy this pipeline wrapper:
   $ hayhooks pipeline deploy -n time_pipeline_toolset integrations/mcp/examples/pipeline_wrapper_toolset.py

```

For more information, see: https://github.com/deepset-ai/hayhooks
"""

from typing import Optional
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from hayhooks import BasePipelineWrapper

from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """
        Setup the pipeline with MCPToolset for time tools.
        
        This creates a pipeline that uses an MCP toolset with time tools
        and then uses the time to answer a user question.
        """
        # Create server info for the time service
        server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

        # Create the toolset - optionally filter to only include specific tools
        mcp_toolset = MCPToolset(
            server_info=server_info,
            tool_names=["get_current_time"]  # Only include the get_current_time tool
        )
        
        # Create the pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=mcp_toolset))
        self.pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
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

    def run_api(self, query: str, timezone: Optional[str] = None) -> str:
        """
        Run the pipeline with a user query.
        
        :param query: The user query asking about time
        :param timezone: Optional timezone to focus on (not directly used but included for API flexibility)
        :return: The response from the LLM
        """
        # Create a user message from the query
        user_input_msg = ChatMessage.from_user(text=query)
        
        # Run the pipeline
        result = self.pipeline.run({
            "llm": {"messages": [user_input_msg]}, 
            "adapter": {"initial_msg": [user_input_msg]}
        })
        
        # Return the text of the first reply
        return result["response_llm"]["replies"][0].text 