# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experimental Agent-configuration optimization APIs for Agent Pack.

Given a reference `Agent`, successful run inputs and outputs, informational model prices, and optimization objectives,
a `HarnessOptimizationExperiment` lets an optimizer Agent edit the full serialized configuration, observe each
measurement, and refine its next choice. Nothing is promoted or deployed automatically.

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
    EvaluationMetrics,
    ModelPrice,
    ModelPriceCatalog,
    ModelTokenUsage,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import (
    AgentMutation,
    MutationOperation,
    OptimizerDecision,
    apply_mutation,
    materialize_mutation,
)
from haystack_integrations.agent_pack.optimization.proposer import (
    HarnessOptimizerAgentProposer,
    MutationProposer,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)

__all__ = [
    "AgentMutation",
    "CandidateEvaluation",
    "EvaluationMetrics",
    "ExperimentJournal",
    "ExperimentRecommendation",
    "ExperimentResult",
    "HarnessEvaluator",
    "HarnessOptimizationExperiment",
    "HarnessOptimizerAgentProposer",
    "ModelPrice",
    "ModelPriceCatalog",
    "ModelTokenUsage",
    "MutationOperation",
    "MutationProposer",
    "OptimizationObjectives",
    "OptimizerDecision",
    "apply_mutation",
    "create_harness_optimizer_agent",
    "create_haystack_documentation_mcp_toolset",
    "materialize_mutation",
]
