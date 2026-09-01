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
from haystack_integrations.agent_pack.optimization.campaign.dataclasses import OptimizationObjectives
from haystack_integrations.agent_pack.optimization.recipes.dataclasses import ModelSubstitutionRecipe
from haystack_integrations.agent_pack.optimization.recipes.serialization import recipe_from_dict
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
        :param objectives: The campaign's gates. Unused: every approved model is offered and gated afterwards.
        :returns: One model substitution per approved alternative.
        """
        reference_model = generator_model_id(generator=reference.chat_generator)
        return [
            ModelSubstitutionRecipe(model_id=model_id)
            for model_id in sorted(assets.models)
            if model_id != reference_model
        ]


def _extract_json_array(text: str) -> list[Any]:
    """
    Read a JSON array out of a model response.

    Configuring structured output on the optimizer's generator is the supported way to get a well-formed reply, see
    `create_harness_optimizer_agent`. This is the fallback for when it is not configured, or when the model wraps the
    array in prose, a fenced code block, or the object a JSON schema requires at the root: the outermost bracket pair
    is extracted rather than letting a formatting slip abort a campaign.

    :param text: The model's reply.
    :returns: The parsed array.
    :raises ValueError: If the reply holds no JSON array.
    :raises json.JSONDecodeError: If the bracketed text is not valid JSON.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        msg = "The response contains no JSON array."
        raise ValueError(msg)
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        msg = "The response does not contain a JSON array."
        raise ValueError(msg)
    return parsed


class HarnessOptimizerAgentProposer:
    """
    Ask a skill-guided Agent for JSON recipes and validate them against the closed recipe schemas.

    Nothing the Agent returns is executed. Every proposal goes through `recipe_from_dict`, so a reply outside the
    typed recipe language is rejected rather than run. Configure structured output on the optimizer Agent's generator
    to keep replies well-formed; see `create_harness_optimizer_agent` for an example.
    """

    def __init__(self, optimizer_agent: Agent, *, max_recipes: int = 8, max_attempts: int = 2) -> None:
        """
        Create an Agent-backed proposer.

        :param optimizer_agent: The Agent that proposes transformations, usually built by
            `create_harness_optimizer_agent`.
        :param max_recipes: How many proposals to accept from one reply.
        :param max_attempts: How many times to ask, including one corrective retry per rejected reply.
        """
        self.optimizer_agent = optimizer_agent
        self.max_recipes = max_recipes
        self.max_attempts = max_attempts

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
        :param objectives: The campaign's gates and ranking preference.
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
            "objectives": objectives.to_dict(),
            "successful_trace_inputs": [
                extract_agent_replay_inputs(artifact=artifact) for artifact in reference_traces[:3]
            ],
            "maximum_recipes": self.max_recipes,
        }

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
        :param objectives: The campaign's gates and ranking preference.
        :returns: The accepted proposals.
        :raises ValueError: If no attempt produced a valid typed recipe array.
        """
        request = self.build_request(
            reference=reference, reference_traces=reference_traces, assets=assets, objectives=objectives
        )
        messages = [ChatMessage.from_user(json.dumps(request, default=str))]
        last_error = ""
        for attempt in range(self.max_attempts):
            result = self.optimizer_agent.run(messages=messages)
            text = result["last_message"].text or ""
            try:
                proposals = _extract_json_array(text=text)
                if len(proposals) > self.max_recipes:
                    msg = f"Return at most {self.max_recipes} recipes."
                    raise ValueError(msg)
                return [recipe_from_dict(data=proposal) for proposal in proposals]
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
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
                        f"That response was rejected: {last_error}. Reply with only a JSON array of supported "
                        "recipe objects, with no prose and no code fences."
                    ),
                ]
        msg = f"The harness optimizer Agent did not return a valid typed recipe array: {last_error}"
        raise ValueError(msg)
