# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Champion/challenger optimization experiments."""

from haystack_integrations.agent_pack.optimization.experiment.dataclasses import (
    CandidateEvaluation,
    EvaluationMetrics,
    ExperimentRecommendation,
    ExperimentResult,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.experiment.experiment import HarnessOptimizationExperiment
from haystack_integrations.agent_pack.optimization.experiment.journal import ExperimentJournal
from haystack_integrations.agent_pack.optimization.experiment.proposers import (
    ApprovedModelRecipeProposer,
    HarnessOptimizerAgentProposer,
)
from haystack_integrations.agent_pack.optimization.experiment.types import HarnessEvaluator, RecipeProposer

__all__ = [
    "ApprovedModelRecipeProposer",
    "CandidateEvaluation",
    "EvaluationMetrics",
    "ExperimentJournal",
    "ExperimentRecommendation",
    "ExperimentResult",
    "HarnessEvaluator",
    "HarnessOptimizationExperiment",
    "HarnessOptimizerAgentProposer",
    "OptimizationObjectives",
    "RecipeProposer",
]
