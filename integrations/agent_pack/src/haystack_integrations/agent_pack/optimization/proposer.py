# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The iterative optimizer Agent and its structured configuration decisions."""

import json
from typing import TYPE_CHECKING, Any, Protocol

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import Toolset

from haystack_integrations.agent_pack.optimization.models import (
    EvaluationMetrics,
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import AgentMutation, OptimizerDecision
from haystack_integrations.agent_pack.runs import AgentRunRecord

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


def _json_compatible(value: Any) -> Any:
    """Convert useful run data to JSON without reducing every rich value to a string."""
    if isinstance(value, ChatMessage):
        return {"role": value.role.value, "text": value.text, "meta": value.meta}
    if isinstance(value, dict):
        return {str(key): _json_compatible(value=item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(value=item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json_compatible(value=value.to_dict())
    return str(value)


class MutationProposer(Protocol):
    """Choose the next candidate after observing all measurements so far."""

    def propose(
        self,
        reference: Agent,
        reference_runs: list[AgentRunRecord],
        pricing: ModelPriceCatalog,
        objectives: OptimizationObjectives,
        baseline: EvaluationMetrics,
        history: list[dict[str, Any]],
    ) -> AgentMutation | None:
        """Return the next mutation, or ``None`` to stop the search."""
        ...


def create_haystack_documentation_mcp_toolset(eager_connect: bool = False) -> "MCPToolset":
    """Create the optional read-only public Haystack documentation toolset."""
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
    max_agent_steps: int = 12,
) -> Agent:
    """Create the Agent that chooses the next configuration experiment."""
    generator = chat_generator or OpenAIResponsesChatGenerator(model="gpt-5.6-sol", timeout=180.0, max_retries=5)
    return Agent(
        chat_generator=generator,
        tools=[docs_toolset] if docs_toolset is not None else None,
        system_prompt=system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )


class HarnessOptimizerAgentProposer:
    """Ask an Agent for one structured mutation after every measured outcome."""

    def __init__(self, optimizer_agent: Agent) -> None:
        """
        Configure the Agent-backed proposer.

        :param optimizer_agent: Agent that receives the reference configuration and experiment history and returns
            the next structured mutation.
        """
        self.optimizer_agent = optimizer_agent

    def build_request(
        self,
        reference: Agent,
        reference_runs: list[AgentRunRecord],
        pricing: ModelPriceCatalog,
        objectives: OptimizationObjectives,
        baseline: EvaluationMetrics,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the complete state the optimizer needs for its next decision."""
        return {
            "reference_agent_configuration": _json_compatible(value=reference.to_dict()),
            "known_model_prices": pricing.to_dict(),
            "objectives": objectives.to_dict(),
            "baseline": baseline.to_dict(),
            "history": history,
            "successful_reference_runs": [
                {
                    "inputs": _json_compatible(value=record.inputs),
                    "outputs": _json_compatible(value=record.outputs),
                }
                for record in reference_runs[:3]
            ],
        }

    def propose(
        self,
        reference: Agent,
        reference_runs: list[AgentRunRecord],
        pricing: ModelPriceCatalog,
        objectives: OptimizationObjectives,
        baseline: EvaluationMetrics,
        history: list[dict[str, Any]],
    ) -> AgentMutation | None:
        """Choose one next mutation through mandatory provider-native structured output."""
        request = self.build_request(
            reference=reference,
            reference_runs=reference_runs,
            pricing=pricing,
            objectives=objectives,
            baseline=baseline,
            history=history,
        )
        result = self.optimizer_agent.run(
            messages=[ChatMessage.from_user(text=json.dumps(request, default=str))],
            generation_kwargs={"text_format": OptimizerDecision},
        )
        text = result["last_message"].text
        if text is None:
            msg = "The harness optimizer Agent returned no structured decision text."
            raise ValueError(msg)
        return OptimizerDecision.model_validate_json(text).mutation


__all__ = [
    "HARNESS_OPTIMIZER_SYSTEM_PROMPT",
    "HarnessOptimizerAgentProposer",
    "MutationProposer",
    "create_harness_optimizer_agent",
    "create_haystack_documentation_mcp_toolset",
]
