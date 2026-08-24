# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Parsing and fingerprinting proposals against the closed recipe language."""

import hashlib
import json
from typing import Any

from haystack_integrations.agent_pack.optimization.recipes.dataclasses import (
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    ToolSelectionRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe

#: Every recipe kind `recipe_from_dict` accepts. Derived from the recipe classes so it cannot drift from the parser.
RECIPE_KINDS = (
    ModelSubstitutionRecipe.kind,
    PromptAndGenerationRecipe.kind,
    ToolSelectionRecipe.kind,
)

#: JSON schema describing a proposal from an optimizer Agent, for configuring structured output on the generator
#: that Agent runs on. It pins the response to an object holding an array of recipe objects with a known `kind`, and
#: deliberately stops there: the per-kind field sets form a union that a strict schema cannot express cleanly, and
#: `recipe_from_dict` is the authoritative validator either way. Constraining the shape is still worth doing, because
#: it removes the prose and code fences that a free-text reply otherwise has to be recovered from.
RECIPE_PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"kind": {"type": "string", "enum": list(RECIPE_KINDS)}},
                "required": ["kind"],
            },
        }
    },
    "required": ["recipes"],
}


def recipe_fingerprint(recipe: CandidateRecipe) -> str:
    """
    Return a stable content hash for a typed candidate recipe.

    :param recipe: The recipe to hash.
    :returns: A hex SHA-256 digest of the recipe's canonical serialized form.
    """
    payload = json.dumps(recipe.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def recipe_from_dict(data: dict[str, Any]) -> CandidateRecipe:
    """
    Parse one optimizer proposal through the closed set of supported recipe schemas.

    This is the boundary that keeps a generated proposal from becoming arbitrary code: anything outside the declared
    recipe kinds is rejected rather than executed. It is also the single place a new transformation is added: a
    recipe dataclass, a branch here, and `RECIPE_KINDS` picks it up.

    :param data: The proposed recipe.
    :returns: The parsed recipe.
    :raises ValueError: If the proposal names an unsupported kind.
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
    msg = f"Unsupported candidate recipe kind: {kind!r}."
    raise ValueError(msg)
