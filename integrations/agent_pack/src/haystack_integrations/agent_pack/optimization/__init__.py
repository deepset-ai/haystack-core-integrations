# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experimental harness-optimization APIs for Agent Pack.

Given a reference `Agent`, recorded inputs and outputs of successful runs, approved models and configuration changes,
and optimization objectives, a `HarnessOptimizationExperiment` lets an optimizer Agent choose a candidate, observe
its measurements, and refine its next choice. It recommends the cheapest or fastest candidate that clears the gate.
Nothing is promoted or deployed automatically: an experiment returns a recommendation for a human to approve.

This API is experimental and may change without a deprecation period.
"""

from haystack_integrations.agent_pack.optimization.experiment import (
    CandidateEvaluation,
    ExperimentJournal,
    ExperimentRecommendation,
    ExperimentResult,
    HarnessEvaluator,
    HarnessOptimizationExperiment,
)
from haystack_integrations.agent_pack.optimization.models import (
    ApprovedAssetCatalog,
    EvaluationMetrics,
    HarnessPatch,
    ModelAsset,
    ModelTokenUsage,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.proposer import (
    HarnessOptimizerAgentProposer,
    RecipeProposer,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.optimization.recipes import (
    ApplyPatchRecipe,
    CandidateRecipe,
    ModelSubstitutionRecipe,
)

__all__ = [
    "ApplyPatchRecipe",
    "ApprovedAssetCatalog",
    "CandidateEvaluation",
    "CandidateRecipe",
    "EvaluationMetrics",
    "ExperimentJournal",
    "ExperimentRecommendation",
    "ExperimentResult",
    "HarnessEvaluator",
    "HarnessOptimizationExperiment",
    "HarnessOptimizerAgentProposer",
    "HarnessPatch",
    "ModelAsset",
    "ModelSubstitutionRecipe",
    "ModelTokenUsage",
    "OptimizationObjectives",
    "RecipeProposer",
    "create_harness_optimizer_agent",
    "create_haystack_documentation_mcp_toolset",
]
