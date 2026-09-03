# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The Agent that chooses optimization experiments and the requests made to it."""

import json
from typing import TYPE_CHECKING, Any

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import Toolset
from haystack.utils import _serialize_value_with_schema

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import AgentMutation, OptimizerDecision

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the Haystack documentation MCP server.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

HARNESS_OPTIMIZER_SYSTEM_PROMPT = """
You optimize a Haystack Agent configuration through a measured sequence of experiments. On every turn you receive
the complete serialized reference Agent configuration, successful reference inputs and outputs, known model prices,
optimization objectives, a baseline measurement, and all candidate outcomes so far. Choose the most informative next
configuration mutation based on that evidence. You may change any part of the serialized Agent configuration and may
combine multiple related changes when that is the best experiment. Return null when no worthwhile experiment remains.

Express edits as ordered RFC 6901 JSON Pointer operations. `set` writes a scalar. `create_object` and `create_array`
create containers that later operations can populate. `remove` deletes a value. `copy` deep-copies any existing
configuration subtree. Array path `-` appends. Escape `~` as `~0` and `/` as `~1` in path segments. Every mutation is
applied to the unchanged reference configuration, not to the preceding candidate. Every operation has `value` and
`from_path` fields: set unused fields to null; only `set` uses `value`, and only `copy` uses `from_path`.

Quality is a hard gate. Optimize the requested primary measurement only among candidates likely to preserve quality.
Known prices are informational rather than an allowlist: you may select other models, but their measured cost cannot be
ranked until pricing is supplied. Use documentation tools before changing an unfamiliar component path or provider
generation argument. Learn from failed mutations and measurements, and do not repeat a resulting configuration.
""".strip()


def create_haystack_documentation_mcp_toolset(eager_connect: bool = False) -> "MCPToolset":
    """
    Create the optional read-only public Haystack documentation toolset.

    :param eager_connect: Connect to the documentation server immediately instead of on first use.
    :returns: A toolset exposing Haystack documentation search.
    """
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url="https://docs.haystack.deepset.ai/api/mcp"),
        tool_names=["search_haystack_docs"],
        eager_connect=eager_connect,
    )


def create_harness_optimizer_agent(
    chat_generator: ChatGenerator | None = None,
    docs_toolset: Toolset | None = None,
    system_prompt: str | None = None,
    additional_instructions: str | None = None,
    max_agent_steps: int = 12,
) -> Agent:
    """
    Create the Agent that chooses the next configuration experiment.

    :param chat_generator: Generator used to reason about experiment results and propose configuration changes.
    :param docs_toolset: Optional Haystack documentation tools available to the Agent.
    :param system_prompt: Optional replacement for the default optimizer instructions.
    :param additional_instructions: Guidance appended to the instructions, for what a good configuration looks like
        in one specific harness. The instructions themselves stay free of any assumption about what the reference
        Agent does, so domain knowledge belongs here rather than in a rewritten replacement.
    :param max_agent_steps: Maximum number of Agent steps used to produce one proposal.
    :returns: The configured optimizer Agent.
    """
    instructions = system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT
    if additional_instructions is not None:
        instructions = f"{instructions}\n\n{additional_instructions.strip()}"
    generator = chat_generator or OpenAIResponsesChatGenerator(model="gpt-5.6-sol", timeout=180.0, max_retries=5)
    return Agent(
        chat_generator=generator,
        tools=[docs_toolset] if docs_toolset is not None else None,
        system_prompt=instructions,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )


def propose_mutation(
    optimizer_agent: Agent,
    reference: Agent,
    reference_runs: list[AgentRunRecord],
    pricing: ModelPriceCatalog,
    objectives: OptimizationObjectives,
    baseline: EvaluationMetrics,
    history: list[dict[str, Any]],
) -> AgentMutation | None:
    """
    Ask the optimizer Agent for the next structured configuration mutation.

    :param optimizer_agent: Agent that chooses the next configuration experiment.
    :param reference: Unchanged reference Agent and source configuration for every candidate.
    :param reference_runs: Reference inputs and outputs that candidates must preserve.
    :param pricing: Known model prices supplied as optimization context.
    :param objectives: Quality gates and primary optimization measurement.
    :param baseline: Measured reference Agent performance.
    :param history: Candidate mutations and outcomes observed so far.
    :returns: The next mutation, or `None` when the optimizer chooses to stop.
    """
    request = {
        "reference_agent_configuration": reference.to_dict(),
        "known_model_prices": pricing.to_dict(),
        "objectives": objectives.to_dict(),
        "baseline": baseline.to_dict(),
        "history": history,
        "successful_reference_runs": [
            {
                "inputs": _serialize_value_with_schema(payload=record.inputs)["serialized_data"],
                "outputs": _serialize_value_with_schema(payload=record.outputs)["serialized_data"],
            }
            for record in reference_runs[:3]
        ],
    }
    result = optimizer_agent.run(
        messages=[ChatMessage.from_user(text=json.dumps(request, default=str))],
        generation_kwargs={"text_format": OptimizerDecision},
    )
    text = result["last_message"].text
    if text is None:
        msg = "The harness optimizer Agent returned no structured decision text."
        raise ValueError(msg)
    return OptimizerDecision.model_validate_json(text).mutation
