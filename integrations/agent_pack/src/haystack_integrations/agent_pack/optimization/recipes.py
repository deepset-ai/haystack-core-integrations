# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Typed, allowlisted Agent transformations and their proposal schema."""

import hashlib
import json
from copy import deepcopy
from typing import Annotated, Any, Literal, TypeAlias

from haystack.components.agents import Agent
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from haystack_integrations.agent_pack.optimization.models import ApprovedAssetCatalog

_LIST_NAME_KEYS = ("data", "init_parameters")


def _serialized_agent(reference: Agent) -> dict[str, Any]:
    try:
        data = reference.to_dict()
    except Exception as error:
        msg = f"{type(reference).__name__} cannot be serialized and optimized: {error}"
        raise ValueError(msg) from error
    if not isinstance(data.get("init_parameters"), dict):
        msg = f"{type(reference).__name__} serialized without initialization parameters."
        raise ValueError(msg)
    return deepcopy(data)


def _named_element(elements: list[Any], name: str) -> dict[str, Any] | None:
    for element in elements:
        if isinstance(element, dict):
            for key in _LIST_NAME_KEYS:
                container = element.get(key)
                if isinstance(container, dict) and container.get("name") == name:
                    return container
    return None


def _resolve_patch_target(data: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    segments = path.split(".")
    current: Any = data
    for index, segment in enumerate(segments[:-1]):
        if isinstance(current, list):
            if (current := _named_element(current, segment)) is None:
                msg = f"Patch path {path!r} names {segment!r}, which is not in the harness."
                raise ValueError(msg)
            continue
        if not isinstance(current, dict):
            reached = ".".join(segments[:index])
            msg = f"Patch path {path!r} cannot traverse the value at {reached!r}."
            raise ValueError(msg)
        current = current.setdefault(segment, {})
    if not isinstance(current, dict):
        msg = f"Patch path {path!r} ends in a value that cannot be traversed."
        raise ValueError(msg)
    return current, segments[-1]


def _patched_agent(reference: Agent, patch: dict[str, Any]) -> Agent:
    data = _serialized_agent(reference)
    parameters = data["init_parameters"]
    for path, value in patch.items():
        container, key = _resolve_patch_target(parameters, path)
        container[key] = value
    try:
        return type(reference).from_dict(data)
    except Exception as error:
        msg = f"The changed harness could not be rebuilt: {error}"
        raise ValueError(msg) from error


class _Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Build the candidate represented by this recipe."""
        raise NotImplementedError

    def identity(self, assets: ApprovedAssetCatalog) -> dict[str, Any]:  # noqa: ARG002
        """Return the recipe plus the complete approved asset it resolves to."""
        return self.model_dump()


class ModelSubstitutionRecipe(_Recipe):
    """Replace the coordinator chat generator with an approved model deployment."""

    kind: Literal["model_substitution"] = "model_substitution"
    model_id: str

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Build an Agent using the selected approved model."""
        data = _serialized_agent(reference)
        patch = assets.model(self.model_id).substitution_patch(
            serialized_generator=data["init_parameters"].get("chat_generator") or {}
        )
        parameters = data["init_parameters"]
        for path, value in patch.items():
            container, key = _resolve_patch_target(parameters, path)
            container[key] = value
        return type(reference).from_dict(data)

    def identity(self, assets: ApprovedAssetCatalog) -> dict[str, Any]:
        """Include the complete model asset in this recipe's measurement identity."""
        return {**self.model_dump(), "asset": assets.model(self.model_id).identity()}


class SystemPromptRecipe(_Recipe):
    """Replace the Agent's system prompt."""

    kind: Literal["system_prompt"] = "system_prompt"
    system_prompt: str

    @field_validator("system_prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "The replacement system prompt cannot be empty."
            raise ValueError(msg)
        return value

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """Build an Agent using the replacement system prompt."""
        return _patched_agent(reference, {"system_prompt": self.system_prompt})


class ApplyPatchRecipe(_Recipe):
    """Apply one named configuration change from the approved catalog."""

    kind: Literal["apply_patch"] = "apply_patch"
    patch: str

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Build an Agent with the selected approved patch applied."""
        return _patched_agent(reference, assets.patch(self.patch).patch)

    def identity(self, assets: ApprovedAssetCatalog) -> dict[str, Any]:
        """Include the complete patch definition in this recipe's measurement identity."""
        return {**self.model_dump(), "asset": assets.patch(self.patch).identity()}


CandidateRecipe: TypeAlias = Annotated[
    ModelSubstitutionRecipe | SystemPromptRecipe | ApplyPatchRecipe,
    Field(discriminator="kind"),
]


class OptimizerProposal(BaseModel):
    """One optimizer decision: the next recipe to evaluate, or ``null`` to stop."""

    model_config = ConfigDict(extra="forbid")
    recipe: CandidateRecipe | None


def proposal_json_schema(assets: ApprovedAssetCatalog) -> dict[str, Any]:
    """Return the proposal schema with model and patch fields closed over the current catalog."""
    schema = OptimizerProposal.model_json_schema()
    definitions = schema.get("$defs", {})
    definitions["ModelSubstitutionRecipe"]["properties"]["model_id"]["enum"] = sorted(assets.models)
    definitions["ApplyPatchRecipe"]["properties"]["patch"]["enum"] = sorted(assets.patches)
    for name in ("ModelSubstitutionRecipe", "SystemPromptRecipe", "ApplyPatchRecipe"):
        definitions[name]["required"] = sorted({*definitions[name].get("required", []), "kind"})
    return schema


def parse_proposal(payload: dict[str, Any], assets: ApprovedAssetCatalog) -> CandidateRecipe | None:
    """Validate one optimizer decision against the typed recipe language and current catalog."""
    raw_recipe = payload.get("recipe")
    if raw_recipe is not None and (not isinstance(raw_recipe, dict) or "kind" not in raw_recipe):
        msg = "A proposed recipe must declare its kind."
        raise ValueError(msg)
    recipe = OptimizerProposal.model_validate(payload).recipe
    if isinstance(recipe, ModelSubstitutionRecipe):
        assets.model(recipe.model_id)
    elif isinstance(recipe, ApplyPatchRecipe):
        assets.patch(recipe.patch)
    return recipe


def recipe_fingerprint(recipe: _Recipe, assets: ApprovedAssetCatalog) -> str:
    """Fingerprint both the recipe and the complete asset definition it selects."""
    payload = json.dumps(recipe.identity(assets), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def is_obvious_noop(recipe: _Recipe, reference: Agent) -> bool:
    """Return whether the recipe visibly asks for the reference's existing value."""
    from haystack_integrations.agent_pack.optimization.models import generator_model_id  # noqa: PLC0415

    if isinstance(recipe, ModelSubstitutionRecipe):
        return recipe.model_id == generator_model_id(reference.chat_generator)
    if isinstance(recipe, SystemPromptRecipe):
        return recipe.system_prompt == reference.system_prompt
    return False


RECIPE_KINDS = ("model_substitution", "system_prompt", "apply_patch")

__all__ = [
    "RECIPE_KINDS",
    "ApplyPatchRecipe",
    "CandidateRecipe",
    "ModelSubstitutionRecipe",
    "SystemPromptRecipe",
    "ValidationError",
    "is_obvious_noop",
    "parse_proposal",
    "proposal_json_schema",
    "recipe_fingerprint",
]
