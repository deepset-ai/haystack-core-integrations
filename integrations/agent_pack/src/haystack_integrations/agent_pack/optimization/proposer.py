# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Propose structured Agent configuration changes from measured outcomes."""

import json
from typing import Any

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.utils import _serialize_value_with_schema

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import AgentMutation, OptimizerDecision


def build_optimizer_request(
    reference: Agent,
    reference_runs: list[AgentRunRecord],
    pricing: ModelPriceCatalog,
    objectives: OptimizationObjectives,
    baseline: EvaluationMetrics,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the complete state the optimizer needs for its next decision.

    :param reference: Unchanged reference Agent and source configuration for every candidate.
    :param reference_runs: Reference inputs and outputs that candidates must preserve.
    :param pricing: Known model prices supplied as optimization context.
    :param objectives: Quality gates and primary optimization measurement.
    :param baseline: Measured reference Agent performance.
    :param history: Candidate mutations and outcomes observed so far.
    :returns: JSON-compatible optimizer decision context.
    """
    return {
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
    request = build_optimizer_request(
        reference=reference,
        reference_runs=reference_runs,
        pricing=pricing,
        objectives=objectives,
        baseline=baseline,
        history=history,
    )
    result = optimizer_agent.run(
        messages=[ChatMessage.from_user(text=json.dumps(request, default=str))],
        generation_kwargs={"text_format": OptimizerDecision},
    )
    text = result["last_message"].text
    if text is None:
        msg = "The harness optimizer Agent returned no structured decision text."
        raise ValueError(msg)
    return OptimizerDecision.model_validate_json(text).mutation


__all__ = [
    "build_optimizer_request",
    "propose_mutation",
]
