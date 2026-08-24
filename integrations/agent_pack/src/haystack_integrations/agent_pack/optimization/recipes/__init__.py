# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Typed, allowlisted transformations for candidate Agent harnesses."""

from haystack_integrations.agent_pack.optimization.recipes.dataclasses import (
    CompositeRecipe,
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    SpecialistDelegationRecipe,
    ToolSelectionRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes.registry import (
    RegisteredStructuralRecipe,
    StructuralRecipeFactory,
    StructuralRecipeRegistry,
)
from haystack_integrations.agent_pack.optimization.recipes.serialization import recipe_fingerprint, recipe_from_dict
from haystack_integrations.agent_pack.optimization.recipes.types import CandidateRecipe

__all__ = [
    "CandidateRecipe",
    "CompositeRecipe",
    "ModelSubstitutionRecipe",
    "PromptAndGenerationRecipe",
    "RegisteredStructuralRecipe",
    "SpecialistDelegationRecipe",
    "StructuralRecipeFactory",
    "StructuralRecipeRegistry",
    "ToolSelectionRecipe",
    "recipe_fingerprint",
    "recipe_from_dict",
]
