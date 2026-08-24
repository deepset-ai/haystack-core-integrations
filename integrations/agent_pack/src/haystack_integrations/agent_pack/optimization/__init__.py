# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experimental harness-optimization APIs for Agent Pack.

Given a reference `Agent`, locally captured traces of successful runs, a catalog of the models and tools a candidate
is allowed to use, and optimization objectives, a `HarnessOptimizationCampaign` evaluates a closed set of typed
transformations and recommends the cheapest or fastest candidate that still clears the quality gate. Nothing is
promoted or deployed automatically: a campaign returns a recommendation for a human to approve.

Compliance is enforced once, at configuration time: `ApprovedAssetCatalog` validates every model and tool a
candidate is built with, including those of delegated agents, and a candidate that uses anything outside the catalog
never runs.

This API is experimental and may change without a deprecation period.
"""

from haystack_integrations.agent_pack.optimization.assets import (
    ApprovedAssetCatalog,
    AssetValidation,
    ModelAsset,
    ToolAsset,
)
from haystack_integrations.agent_pack.optimization.campaign import (
    ApprovedModelRecipeProposer,
    CampaignJournal,
    CampaignRecommendation,
    CampaignResult,
    CandidateEvaluation,
    EvaluationMetrics,
    HarnessEvaluator,
    HarnessOptimizationCampaign,
    HarnessOptimizerAgentProposer,
    OptimizationObjectives,
    RecipeProposer,
)
from haystack_integrations.agent_pack.optimization.optimizer_agent import (
    create_harness_optimizer_agent,
    create_haystack_docs_toolset,
)
from haystack_integrations.agent_pack.optimization.recipes import (
    CandidateRecipe,
    CompositeRecipe,
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    RegisteredStructuralRecipe,
    SpecialistDelegationRecipe,
    StructuralRecipeRegistry,
    ToolSelectionRecipe,
)
from haystack_integrations.agent_pack.optimization.tracing import (
    LocalTraceCollector,
    LocalTraceStore,
    TraceArtifact,
    TraceCaptureLimits,
    TraceCapturingAgentRunner,
    TraceSelection,
    TraceSource,
    span_tag,
)

__all__ = [
    "ApprovedAssetCatalog",
    "ApprovedModelRecipeProposer",
    "AssetValidation",
    "CampaignJournal",
    "CampaignRecommendation",
    "CampaignResult",
    "CandidateEvaluation",
    "CandidateRecipe",
    "CompositeRecipe",
    "EvaluationMetrics",
    "HarnessEvaluator",
    "HarnessOptimizationCampaign",
    "HarnessOptimizerAgentProposer",
    "LocalTraceCollector",
    "LocalTraceStore",
    "ModelAsset",
    "ModelSubstitutionRecipe",
    "OptimizationObjectives",
    "PromptAndGenerationRecipe",
    "RecipeProposer",
    "RegisteredStructuralRecipe",
    "SpecialistDelegationRecipe",
    "StructuralRecipeRegistry",
    "ToolAsset",
    "ToolSelectionRecipe",
    "TraceArtifact",
    "TraceCaptureLimits",
    "TraceCapturingAgentRunner",
    "TraceSelection",
    "TraceSource",
    "create_harness_optimizer_agent",
    "create_haystack_docs_toolset",
    "span_tag",
]
