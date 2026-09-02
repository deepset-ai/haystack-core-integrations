# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Typed, allowlisted transformations for candidate Agent harnesses."""

from haystack_integrations.agent_pack.optimization.recipes.dataclasses import (
    ApplyPatchRecipe,
    ModelSubstitutionRecipe,
    SystemPromptRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes.proposals import (
    RECIPE_KINDS,
    parse_proposal,
    proposal_json_schema,
)
from haystack_integrations.agent_pack.optimization.recipes.serialization import recipe_fingerprint
from haystack_integrations.agent_pack.optimization.recipes.types import CandidateRecipe

__all__ = [
    "RECIPE_KINDS",
    "ApplyPatchRecipe",
    "CandidateRecipe",
    "ModelSubstitutionRecipe",
    "SystemPromptRecipe",
    "parse_proposal",
    "proposal_json_schema",
    "recipe_fingerprint",
]
