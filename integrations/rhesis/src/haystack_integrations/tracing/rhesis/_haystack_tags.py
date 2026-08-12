# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
The Haystack span-tag and operation names this integration matches on.

One source of truth on purpose: getting these literals right is the package's whole job, and they
were previously declared independently in both the tracer and the mapping layer.
"""

from __future__ import annotations

# Operations, as Haystack names them when it opens a span.
PIPELINE_RUN = "haystack.pipeline.run"
ASYNC_PIPELINE_RUN = "haystack.async_pipeline.run"
COMPONENT_RUN = "haystack.component.run"
AGENT_RUN = "haystack.agent.run"

# Haystack 3.0 moved the Agent loop off ``Pipeline._run_component``: each iteration now opens its own
# span, and the LLM call and every tool call are traced directly instead of through a ``ToolInvoker``
# component span. These operations carry no ``haystack.component.name``/``.type`` tags, so they are
# matched on operation name alone.
AGENT_STEP = "haystack.agent.step"
AGENT_STEP_LLM = "haystack.agent.step.llm"
AGENT_STEP_TOOL = "haystack.agent.step.tool"

# Tags carried on those spans.
PIPELINE_INPUT = "haystack.pipeline.input_data"
PIPELINE_OUTPUT = "haystack.pipeline.output_data"
COMPONENT_NAME = "haystack.component.name"
COMPONENT_TYPE = "haystack.component.type"
COMPONENT_INPUT = "haystack.component.input"
COMPONENT_OUTPUT = "haystack.component.output"
AGENT_INPUT = "haystack.agent.input"
AGENT_OUTPUT = "haystack.agent.output"
AGENT_STEP_LLM_OUTPUT = "haystack.agent.step.llm.output"
AGENT_STEP_TOOL_INPUT = "haystack.agent.step.tool.input"
AGENT_STEP_TOOL_OUTPUT = "haystack.agent.step.tool.output"
TOOL_NAME = "haystack.tool.name"
TOOL_DESCRIPTION = "haystack.tool.description"
