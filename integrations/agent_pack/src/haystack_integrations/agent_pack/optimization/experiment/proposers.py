# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Where candidate recipes come from."""

import json
from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.assets.model_identity import generator_model_id
from haystack_integrations.agent_pack.optimization.experiment.dataclasses import OptimizationObjectives
from haystack_integrations.agent_pack.optimization.recipes.dataclasses import ModelSubstitutionRecipe
from haystack_integrations.agent_pack.optimization.recipes.proposals import (
    ValidationError,
    parse_proposal,
    proposal_json_schema,
)
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe
from haystack_integrations.agent_pack.tracing.dataclasses import TraceArtifact
from haystack_integrations.agent_pack.tracing.extraction import extract_agent_replay_inputs

logger = logging.getLogger(__name__)


class ApprovedModelRecipeProposer:
    """Deterministically evaluate every approved alternative model."""

    def propose(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],  # noqa: ARG002
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,  # noqa: ARG002
    ) -> list[CandidateRecipe]:
        """
        Return one substitution recipe per approved non-reference model.

        :param reference: The champion harness, read for the model it already uses.
        :param reference_traces: The selected reference traces. Unused: this proposer does not read run content.
        :param assets: The approved model and tool allowlist.
        :param objectives: The experiment's gates. Unused: every approved model is offered and gated afterwards.
        :returns: One model substitution per approved alternative.
        """
        reference_model = generator_model_id(generator=reference.chat_generator)
        return [
            ModelSubstitutionRecipe(model_id=model_id)
            for model_id in sorted(assets.models)
            if model_id != reference_model
        ]


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Read a JSON object out of a model response.

    Structured output is the supported way to get a well-formed reply, and `structured_output_key` configures it.
    This is the fallback for when it is not configured, or when the model wraps the object in prose or a fenced code
    block: the outermost brace pair is extracted rather than letting a formatting slip abort an experiment.

    :param text: The model's reply.
    :returns: The parsed object.
    :raises ValueError: If the reply holds no JSON object.
    :raises json.JSONDecodeError: If the braced text is not valid JSON.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        msg = "The response contains no JSON object."
        raise ValueError(msg)
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        msg = "The response does not contain a JSON object."
        raise ValueError(msg)
    return parsed


class HarnessOptimizerAgentProposer:
    """
    Ask a skill-guided Agent for JSON recipes and validate them against the closed recipe schemas.

    Nothing the Agent returns is executed. Every reply is validated against a Pydantic model generated from the
    approved asset catalog, in which the model IDs and tool names are closed choices, so a reply naming anything
    outside the catalog is rejected here and never reaches a harness.

    Set `structured_output_key` to have that same schema configured on the generator, which keeps replies
    well-formed rather than relying on recovery from free text.
    """

    def __init__(
        self,
        optimizer_agent: Agent,
        *,
        max_recipes: int = 8,
        max_attempts: int = 2,
        structured_output_key: str | None = None,
    ) -> None:
        """
        Create an Agent-backed proposer.

        :param optimizer_agent: The Agent that proposes transformations, usually built by
            `create_harness_optimizer_agent`.
        :param max_recipes: How many proposals to accept from one reply.
        :param max_attempts: How many times to ask, including one corrective retry per rejected reply.
        :param structured_output_key: The generation parameter the generator takes a response schema under, so the
            schema generated from the catalog can be configured per request. `text` for
            `OpenAIResponsesChatGenerator`, `response_format` for `OpenAIChatGenerator`. Left unset, no structured
            output is configured and replies are recovered from free text; validation is unchanged either way.
        """
        self.optimizer_agent = optimizer_agent
        self.max_recipes = max_recipes
        self.max_attempts = max_attempts
        self.structured_output_key = structured_output_key

    def build_request(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
    ) -> dict[str, Any]:
        """
        Describe the optimization task for the optimizer Agent.

        :param reference: The champion harness.
        :param reference_traces: The selected reference traces, sampled for representative inputs.
        :param assets: The approved model and tool allowlist.
        :param objectives: The experiment's gates and ranking preference.
        :returns: The JSON-compatible request payload.
        """
        return {
            "reference": {
                "model": generator_model_id(generator=reference.chat_generator),
                "system_prompt": reference.system_prompt,
                "tools": [configured.name for configured in flatten_tools_or_toolsets(tools=reference.tools)],
            },
            "approved_models": [
                {
                    "model_id": asset.model_id,
                    "provider": asset.provider,
                    "deployment": asset.deployment,
                    "input_cost_per_million": asset.input_cost_per_million,
                    "output_cost_per_million": asset.output_cost_per_million,
                }
                for asset in assets.models.values()
            ],
            "approved_tools": sorted(assets.tools),
            "approved_patches": [
                {"name": declared.name, "description": declared.description, "changes": sorted(declared.patch)}
                for declared in assets.patches.values()
            ],
            "objectives": objectives.to_dict(),
            "successful_trace_inputs": [
                extract_agent_replay_inputs(artifact=artifact) for artifact in reference_traces[:3]
            ],
            "maximum_recipes": self.max_recipes,
        }

    def _structured_output(self, *, assets: ApprovedAssetCatalog) -> dict[str, Any] | None:
        """Build the generation parameter that constrains the reply to the schema generated from the catalog."""
        if self.structured_output_key is None:
            return None
        schema = {
            "type": "json_schema",
            "name": "harness_optimizer_proposal",
            "schema": proposal_json_schema(assets=assets, max_recipes=self.max_recipes),
            # Not strict: the per-kind field sets form a union a strict schema cannot express cleanly, and the
            # Pydantic model remains the authoritative validator.
            "strict": False,
        }
        if self.structured_output_key == "text":
            return {"text": {"format": schema}}
        return {self.structured_output_key: {"type": "json_schema", "json_schema": schema}}

    def propose(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
    ) -> list[CandidateRecipe]:
        """
        Request proposals and reject any response outside the typed recipe language.

        :param reference: The champion harness.
        :param reference_traces: The selected reference traces.
        :param assets: The approved model and tool allowlist.
        :param objectives: The experiment's gates and ranking preference.
        :returns: The accepted proposals.
        :raises ValueError: If no attempt produced a valid typed recipe array.
        """
        request = self.build_request(
            reference=reference, reference_traces=reference_traces, assets=assets, objectives=objectives
        )
        messages = [ChatMessage.from_user(json.dumps(request, default=str))]
        generation_kwargs = self._structured_output(assets=assets)
        last_error = ""
        for attempt in range(self.max_attempts):
            result = self.optimizer_agent.run(messages=messages, generation_kwargs=generation_kwargs)
            text = result["last_message"].text or ""
            try:
                payload = _extract_json_object(text=text)
                return parse_proposal(payload=payload, assets=assets, max_recipes=self.max_recipes)
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                last_error = f"{type(error).__name__}: {error}"
                logger.warning(
                    "Harness optimizer proposal attempt {attempt} was rejected: {error}",
                    attempt=attempt + 1,
                    error=last_error,
                )
                messages = [
                    *messages,
                    ChatMessage.from_assistant(text),
                    ChatMessage.from_user(
                        f"That response was rejected: {last_error}. Reply with only a JSON object holding a "
                        "'recipes' array of supported recipe objects, with no prose and no code fences."
                    ),
                ]
        msg = f"The harness optimizer Agent did not return a valid proposal: {last_error}"
        raise ValueError(msg)
