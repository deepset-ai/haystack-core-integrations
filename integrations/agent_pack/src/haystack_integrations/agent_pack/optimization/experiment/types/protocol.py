# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocols for the pluggable parts of an experiment."""

from typing import Any, Protocol

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.experiment.dataclasses import (
    EvaluationMetrics,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe
from haystack_integrations.agent_pack.tracing.dataclasses import TraceArtifact


class HarnessEvaluator(Protocol):
    """Evaluate one materialized candidate against selected reference traces."""

    def evaluate(
        self, agent: Agent, reference_traces: list[TraceArtifact], assets: ApprovedAssetCatalog
    ) -> EvaluationMetrics:
        """
        Return comparable experiment metrics.

        :param agent: The materialized candidate to score.
        :param reference_traces: The reference traces to evaluate against.
        :param assets: The approved asset catalog, supplied so cost is priced from the same catalog the experiment
            gates against instead of a second price table that can drift out of step with it.
        :returns: Quality, cost, and latency for the candidate.
        """
        ...

    def fingerprint(self) -> dict[str, Any]:
        """
        Return a JSON-compatible description of everything that affects this evaluator's scores.

        Folded into the experiment's configuration hash so a change to the evaluation set, the repetition count, or
        the price table invalidates journaled results instead of silently reusing them. Optional.

        :returns: The evaluator's identifying configuration.
        """
        ...


class RecipeProposer(Protocol):
    """Generate typed candidate recipes from approved inputs."""

    def propose(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
    ) -> list[CandidateRecipe]:
        """
        Return a finite list of constrained recipes.

        :param reference: The Agent being optimized.
        :param reference_traces: The selected reference traces.
        :param assets: The approved model and tool allowlist.
        :param objectives: The experiment's gates and ranking preference.
        :returns: The proposed candidate transformations.
        """
        ...
