# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Parsing and fingerprinting proposals against the closed recipe language."""

import hashlib
import json
from typing import Any

from haystack_integrations.agent_pack.optimization.recipes.dataclasses import (
    CompositeRecipe,
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    SpecialistDelegationRecipe,
    ToolSelectionRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes.registry import (
    RegisteredStructuralRecipe,
    StructuralRecipeRegistry,
)
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe


def recipe_fingerprint(recipe: CandidateRecipe) -> str:
    """
    Return a stable content hash for a typed candidate recipe.

    :param recipe: The recipe to hash.
    :returns: A hex SHA-256 digest of the recipe's canonical serialized form.
    """
    payload = json.dumps(recipe.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def recipe_from_dict(data: dict[str, Any], *, registry: StructuralRecipeRegistry | None = None) -> CandidateRecipe:
    """
    Parse one optimizer proposal through the closed set of supported recipe schemas.

    This is the boundary that keeps a generated proposal from becoming arbitrary code: anything outside the declared
    recipe kinds is rejected rather than executed.

    :param data: The proposed recipe.
    :param registry: The registry to resolve a `registered_structure` recipe against.
    :returns: The parsed recipe.
    :raises ValueError: If the proposal names an unsupported kind, or a registered structure with no registry.
    :raises KeyError: If a required field for the named kind is missing.
    """
    kind = data.get("kind")
    if kind == ModelSubstitutionRecipe.kind:
        return ModelSubstitutionRecipe(model_id=data["model_id"])
    if kind == PromptAndGenerationRecipe.kind:
        return PromptAndGenerationRecipe(
            system_prompt=data.get("system_prompt"), generation_kwargs=data.get("generation_kwargs")
        )
    if kind == ToolSelectionRecipe.kind:
        return ToolSelectionRecipe(tool_names=tuple(data["tool_names"]))
    if kind == SpecialistDelegationRecipe.kind:
        return SpecialistDelegationRecipe(
            name=data["name"],
            description=data["description"],
            specialist_tool_names=tuple(data["specialist_tool_names"]),
            specialist_system_prompt=data["specialist_system_prompt"],
            coordinator_tool_names=tuple(data.get("coordinator_tool_names") or ()),
            specialist_model_id=data.get("specialist_model_id"),
            coordinator_system_prompt=data.get("coordinator_system_prompt"),
        )
    if kind == CompositeRecipe.kind:
        return CompositeRecipe(
            recipes=tuple(recipe_from_dict(data=item, registry=registry) for item in data["recipes"])
        )
    if kind == RegisteredStructuralRecipe.kind:
        if registry is None:
            msg = "A StructuralRecipeRegistry is required for registered structural recipes."
            raise ValueError(msg)
        return RegisteredStructuralRecipe(name=data["name"], parameters=data.get("parameters") or {}, registry=registry)
    msg = f"Unsupported candidate recipe kind: {kind!r}."
    raise ValueError(msg)
