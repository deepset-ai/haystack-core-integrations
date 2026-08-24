# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Champion/challenger optimization campaigns."""

from haystack_integrations.agent_pack.optimization.campaign.campaign import HarnessOptimizationCampaign
from haystack_integrations.agent_pack.optimization.campaign.dataclasses import (
    CampaignRecommendation,
    CampaignResult,
    CandidateEvaluation,
    EvaluationMetrics,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.campaign.journal import CampaignJournal
from haystack_integrations.agent_pack.optimization.campaign.proposers import (
    ApprovedModelRecipeProposer,
    HarnessOptimizerAgentProposer,
)
from haystack_integrations.agent_pack.optimization.campaign.types import HarnessEvaluator, RecipeProposer

__all__ = [
    "ApprovedModelRecipeProposer",
    "CampaignJournal",
    "CampaignRecommendation",
    "CampaignResult",
    "CandidateEvaluation",
    "EvaluationMetrics",
    "HarnessEvaluator",
    "HarnessOptimizationCampaign",
    "HarnessOptimizerAgentProposer",
    "OptimizationObjectives",
    "RecipeProposer",
]
