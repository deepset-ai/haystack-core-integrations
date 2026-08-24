# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Typed, allowlisted transformations for candidate Agent harnesses."""

from haystack_integrations.agent_pack.optimization.recipes.dataclasses import (
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    ToolSelectionRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes.serialization import (
    RECIPE_KINDS,
    RECIPE_PROPOSAL_JSON_SCHEMA,
    recipe_fingerprint,
    recipe_from_dict,
)
from haystack_integrations.agent_pack.optimization.recipes.types import CandidateRecipe

__all__ = [
    "RECIPE_KINDS",
    "RECIPE_PROPOSAL_JSON_SCHEMA",
    "CandidateRecipe",
    "ModelSubstitutionRecipe",
    "PromptAndGenerationRecipe",
    "ToolSelectionRecipe",
    "recipe_fingerprint",
    "recipe_from_dict",
]
