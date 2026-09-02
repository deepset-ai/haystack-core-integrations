# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The schema an optimizer answers against, generated from the approved asset catalog."""

from functools import reduce
from operator import or_
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.recipes.dataclasses import (
    ApplyPatchRecipe,
    ModelSubstitutionRecipe,
    SystemPromptRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe

#: Every transformation an optimizer may propose. Derived from the recipe classes, so it cannot drift from them.
RECIPE_KINDS = (ModelSubstitutionRecipe.kind, SystemPromptRecipe.kind, ApplyPatchRecipe.kind)


def _build_proposal_model(*, assets: ApprovedAssetCatalog, max_recipes: int) -> type[BaseModel]:
    """
    Build the Pydantic model an optimizer's reply is validated against.

    The model is generated per catalog rather than written out once, because the approved model IDs and tool names
    become closed choices inside it. That is what makes the catalog the compliance boundary: a proposal naming
    anything outside it cannot be validated, so it never reaches a harness and nothing needs re-checking afterwards.

    :param assets: The catalog whose model IDs and tool names become the allowed choices.
    :param max_recipes: How many proposals one reply may contain.
    :returns: A model with a `recipes` list of discriminated recipe proposals.
    """
    strict = ConfigDict(extra="forbid")

    class SystemPromptProposal(BaseModel):
        """Replace the harness's system prompt."""

        model_config = strict
        kind: Literal["system_prompt"]
        system_prompt: str = Field(min_length=1)

    # Only the kinds the catalog can support are offered: with no approved models there is nothing to substitute,
    # and with no approved patches there is no configuration change to apply. A dynamic `Literal` renders as a JSON
    # Schema `enum`, so the closed choices reach a structured-output schema as well as Pydantic's validator.
    members: list[type[BaseModel]] = [SystemPromptProposal]

    if assets.models:
        model_choice = Literal[tuple(sorted(assets.models))]  # type: ignore[valid-type]

        class ModelSubstitutionProposal(BaseModel):
            """Swap the coordinator model for an approved one."""

            model_config = strict
            kind: Literal["model_substitution"]
            model_id: model_choice  # type: ignore[valid-type]

        members.append(ModelSubstitutionProposal)

    if assets.patches:
        patch_choice = Literal[tuple(sorted(assets.patches))]  # type: ignore[valid-type]

        class ApplyPatchProposal(BaseModel):
            """Apply one approved configuration change."""

            model_config = strict
            kind: Literal["apply_patch"]
            patch: patch_choice  # type: ignore[valid-type]

        members.append(ApplyPatchProposal)

    proposal: Any = Annotated[reduce(or_, members), Field(discriminator="kind")] if len(members) > 1 else members[0]

    class HarnessProposal(BaseModel):
        """One optimizer reply: the transformations it suggests trying."""

        model_config = strict
        # An empty list is valid: having nothing worth trying is a legitimate answer.
        recipes: list[proposal] = Field(default_factory=list, max_length=max_recipes)  # type: ignore[valid-type]

    return HarnessProposal


def proposal_json_schema(*, assets: ApprovedAssetCatalog, max_recipes: int = 8) -> dict[str, Any]:
    """
    Return the JSON schema for a proposal, for configuring structured output on an optimizer's generator.

    :param assets: The catalog whose model IDs and tool names become the allowed choices.
    :param max_recipes: How many proposals one reply may contain.
    :returns: A JSON schema describing one reply.
    """
    return _build_proposal_model(assets=assets, max_recipes=max_recipes).model_json_schema()


def parse_proposal(
    *, payload: dict[str, Any], assets: ApprovedAssetCatalog, max_recipes: int = 8
) -> list[CandidateRecipe]:
    """
    Validate one optimizer reply and convert it into recipes.

    This is the boundary that keeps a generated proposal from becoming arbitrary change: an unknown transformation,
    an unapproved model or tool, an unexpected field, or too many proposals all fail here.

    :param payload: The reply to validate.
    :param assets: The catalog whose model IDs and tool names became the allowed choices.
    :param max_recipes: How many proposals one reply may contain.
    :returns: The proposed transformations.
    :raises ValidationError: If the reply does not match the schema generated from the catalog.
    """
    model = _build_proposal_model(assets=assets, max_recipes=max_recipes)
    validated = model.model_validate(payload)
    # The model is generated, so its fields are reached by name rather than by a static type.
    proposals: list[BaseModel] = validated.recipes  # type: ignore[attr-defined]
    return [_recipe_from_proposal(proposal=proposal) for proposal in proposals]


def _recipe_from_proposal(*, proposal: BaseModel) -> CandidateRecipe:
    """Convert one validated proposal into the recipe that applies it."""
    data = proposal.model_dump()
    kind = data["kind"]
    if kind == ModelSubstitutionRecipe.kind:
        return ModelSubstitutionRecipe(model_id=data["model_id"])
    if kind == SystemPromptRecipe.kind:
        return SystemPromptRecipe(system_prompt=data["system_prompt"])
    return ApplyPatchRecipe(patch=data["patch"])


__all__ = ["RECIPE_KINDS", "ValidationError", "parse_proposal", "proposal_json_schema"]
