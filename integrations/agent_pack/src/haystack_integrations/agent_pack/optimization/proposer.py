# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The iterative optimizer Agent and its typed recipe decisions."""

import json
from typing import TYPE_CHECKING, Any, Protocol

from haystack import logging
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import Toolset, flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.models import (
    ApprovedAssetCatalog,
    EvaluationMetrics,
    OptimizationObjectives,
    generator_model_id,
)
from haystack_integrations.agent_pack.optimization.recipes import (
    CandidateRecipe,
    ValidationError,
    parse_proposal,
    proposal_json_schema,
)
from haystack_integrations.agent_pack.runs import AgentRunRecord

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the Haystack documentation MCP server.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

logger = logging.getLogger(__name__)

HARNESS_OPTIMIZER_SYSTEM_PROMPT = """
You optimize a Haystack Agent harness through a measured sequence of experiments. On every turn you receive the
reference harness, approved models and configuration patches, optimization objectives, a baseline measurement, and
the complete history of recipes already evaluated. Choose the single most informative next recipe based on those
observed outcomes. Return null when the evidence says no remaining change is worth evaluating.

Rules:
1. Use only model IDs in `approved_models` and patch names in `approved_patches`.
2. Propose exactly one change at a time so its effect is attributable. Multi-change candidates are not supported.
3. Never repeat a recipe in `history` or propose the reference model.
4. Quality is a hard gate. Optimize the requested primary measurement only among candidates likely to preserve it.
5. Never emit Python, component dictionaries, import paths, credentials, or deployment operations.
6. Reply with one JSON object: `{"recipe": <recipe>}` or `{"recipe": null}`. No prose or code fences.

Supported recipes:
- `{"kind": "model_substitution", "model_id": "approved-model-id"}`
- `{"kind": "apply_patch", "patch": "approved-patch-name"}`
""".strip()


def _json_compatible(value: Any) -> Any:
    """Preserve useful run inputs in the optimizer request instead of reducing rich values to strings."""
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


class RecipeProposer(Protocol):
    """Choose the next candidate after observing all measurements so far."""

    def propose(
        self,
        *,
        reference: Agent,
        reference_runs: list[AgentRunRecord],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
        baseline: EvaluationMetrics,
        history: list[dict[str, Any]],
    ) -> CandidateRecipe | None:
        """Return the next recipe, or ``None`` to stop the search."""
        ...


def create_haystack_documentation_mcp_toolset(*, eager_connect: bool = False) -> "MCPToolset":
    """Create the optional read-only public Haystack documentation toolset."""
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url="https://docs.haystack.deepset.ai/api/mcp"),
        tool_names=["search_haystack_docs"],
        eager_connect=eager_connect,
    )


def create_harness_optimizer_agent(
    *,
    chat_generator: ChatGenerator | None = None,
    docs_toolset: Toolset | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 12,
) -> Agent:
    """Create the Agent that chooses the next harness experiment."""
    generator = chat_generator or OpenAIResponsesChatGenerator(model="gpt-5.6-sol", timeout=180.0, max_retries=5)
    return Agent(
        chat_generator=generator,
        tools=[docs_toolset] if docs_toolset is not None else None,
        system_prompt=system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the outermost JSON object from an optimizer reply."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        msg = "The optimizer response contains no JSON object."
        raise ValueError(msg)
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        msg = "The optimizer response is not a JSON object."
        raise ValueError(msg)
    return parsed


class HarnessOptimizerAgentProposer:
    """Ask an Agent for one next recipe, with measured history supplied on every request."""

    def __init__(
        self,
        optimizer_agent: Agent,
        *,
        max_attempts: int = 2,
        structured_output_key: str | None = None,
    ) -> None:
        """Configure the Agent-backed proposer and its response validation retries."""
        self.optimizer_agent = optimizer_agent
        self.max_attempts = max_attempts
        self.structured_output_key = structured_output_key

    def build_request(
        self,
        *,
        reference: Agent,
        reference_runs: list[AgentRunRecord],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
        baseline: EvaluationMetrics,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the complete state the optimizer needs for its next decision."""
        return {
            "reference": {
                "model": generator_model_id(generator=reference.chat_generator),
                "system_prompt": reference.system_prompt,
                "tools": [
                    {"name": tool.name, "description": tool.description}
                    for tool in flatten_tools_or_toolsets(tools=reference.tools)
                ],
            },
            "approved_models": [
                {
                    "model_id": asset.model_id,
                    "input_cost_per_million": asset.input_cost_per_million,
                    "output_cost_per_million": asset.output_cost_per_million,
                }
                for asset in assets.models.values()
            ],
            "approved_patches": [
                {
                    "name": patch.name,
                    "description": patch.description,
                    "changes": _json_compatible(value=patch.patch),
                }
                for patch in assets.patches.values()
            ],
            "objectives": objectives.to_dict(),
            "baseline": baseline.to_dict(),
            "history": history,
            "successful_run_inputs": [_json_compatible(value=record.inputs) for record in reference_runs[:3]],
        }

    def _structured_output(self, assets: ApprovedAssetCatalog) -> dict[str, Any] | None:
        """Build provider-specific structured-output arguments when configured."""
        if self.structured_output_key is None:
            return None
        schema = {
            "type": "json_schema",
            "name": "harness_optimizer_decision",
            "schema": proposal_json_schema(assets=assets),
            "strict": False,
        }
        if self.structured_output_key == "text":
            return {"text": {"format": schema}}
        return {self.structured_output_key: {"type": "json_schema", "json_schema": schema}}

    def propose(
        self,
        *,
        reference: Agent,
        reference_runs: list[AgentRunRecord],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
        baseline: EvaluationMetrics,
        history: list[dict[str, Any]],
    ) -> CandidateRecipe | None:
        """Choose one next recipe after seeing baseline and prior candidate outcomes."""
        request = self.build_request(
            reference=reference,
            reference_runs=reference_runs,
            assets=assets,
            objectives=objectives,
            baseline=baseline,
            history=history,
        )
        messages = [ChatMessage.from_user(text=json.dumps(request, default=str))]
        last_error = ""
        for attempt in range(self.max_attempts):
            result = self.optimizer_agent.run(
                messages=messages,
                generation_kwargs=self._structured_output(assets),
            )
            text = result["last_message"].text or ""
            try:
                return parse_proposal(payload=_extract_json_object(text=text), assets=assets)
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                last_error = f"{type(error).__name__}: {error}"
                logger.warning(
                    "Harness optimizer decision attempt {attempt} was rejected: {error}",
                    attempt=attempt + 1,
                    error=last_error,
                )
                messages.extend(
                    [
                        ChatMessage.from_assistant(text=text),
                        ChatMessage.from_user(
                            text=(
                                f"That decision was rejected: {last_error}. Reply only with "
                                '{"recipe": <supported recipe or null>}.'
                            )
                        ),
                    ]
                )
        msg = f"The harness optimizer Agent did not return a valid decision: {last_error}"
        raise ValueError(msg)


__all__ = [
    "HARNESS_OPTIMIZER_SYSTEM_PROMPT",
    "HarnessOptimizerAgentProposer",
    "RecipeProposer",
    "create_harness_optimizer_agent",
    "create_haystack_documentation_mcp_toolset",
]
