# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experimental harness-optimization APIs for Agent Pack.

Given a reference `Agent`, locally captured traces of successful runs, a catalog of the models and tools a candidate
is allowed to use, and optimization objectives, a `HarnessOptimizationExperiment` evaluates a closed set of typed
transformations and recommends the cheapest or fastest candidate that still clears the quality gate. Nothing is
promoted or deployed automatically: an experiment returns a recommendation for a human to approve.

Traces come from `haystack_integrations.agent_pack.tracing`, which is a standalone module: import the collector and
the store from there.

Compliance is enforced at the proposal boundary: the schema an optimizer answers against is generated from
`ApprovedAssetCatalog`, so a proposal naming a model, or a configuration change, outside the catalog fails
validation and never reaches a harness.

This API is experimental and may change without a deprecation period.
"""

from haystack_integrations.agent_pack.optimization.assets import (
    ApprovedAssetCatalog,
    HarnessPatch,
    ModelAsset,
    ToolAsset,
)
from haystack_integrations.agent_pack.optimization.experiment import (
    ApprovedModelRecipeProposer,
    CandidateEvaluation,
    EvaluationMetrics,
    ExperimentJournal,
    ExperimentRecommendation,
    ExperimentResult,
    HarnessEvaluator,
    HarnessOptimizationExperiment,
    HarnessOptimizerAgentProposer,
    OptimizationObjectives,
    RecipeProposer,
)
from haystack_integrations.agent_pack.optimization.optimizer_agent import (
    create_harness_optimizer_agent,
    create_haystack_docs_toolset,
)
from haystack_integrations.agent_pack.optimization.recipes import (
    ApplyPatchRecipe,
    CandidateRecipe,
    ModelSubstitutionRecipe,
    SystemPromptRecipe,
)

__all__ = [
    "ApplyPatchRecipe",
    "ApprovedAssetCatalog",
    "ApprovedModelRecipeProposer",
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
    "OptimizationObjectives",
    "RecipeProposer",
    "SystemPromptRecipe",
    "ToolAsset",
    "create_harness_optimizer_agent",
    "create_haystack_docs_toolset",
]
