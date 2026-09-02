# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The closed set of typed, allowlisted transformations for candidate Agent harnesses."""

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.recipes.utils import (
    _patched_agent,
    _rebuild_harness,
    _serialized_harness,
)


@dataclass(frozen=True, kw_only=True)
class ModelSubstitutionRecipe:
    """
    Replace the coordinator model with an approved model deployment.

    Expressed as a patch like every other change; the asset supplies it, so the operator declares a model once with
    its prices rather than writing a patch per model.

    :param model_id: The approved model to switch to.
    """

    model_id: str
    kind: ClassVar[str] = "model_substitution"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Rebuild the Agent with the selected model asset's generator configuration.

        :param reference: The Agent being optimized. Never modified; a candidate is rebuilt from its configuration.
        :param assets: The approved model allowlist.
        :returns: The new candidate Agent.
        :raises ValueError: If the model is not approved, or the harness cannot be serialized or rebuilt.
        """
        data = _serialized_harness(reference=reference)
        patch = assets.model(model_id=self.model_id).substitution_patch(
            serialized_generator=data["init_parameters"].get("chat_generator") or {}
        )
        return _rebuild_harness(reference=reference, data=data, patch=patch)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ModelSubstitutionRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'model_id'.
        """
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True, kw_only=True)
class SystemPromptRecipe:
    """
    Replace the harness's system prompt.

    The only transformation whose value an optimizer authors rather than picks, because a prompt cannot be
    enumerated in advance.

    :param system_prompt: The replacement prompt.
    """

    system_prompt: str
    kind: ClassVar[str] = "system_prompt"

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            msg = "SystemPromptRecipe requires a non-empty prompt."
            raise ValueError(msg)

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """
        Rebuild the Agent with the replacement prompt.

        :param reference: The Agent being optimized. Never modified; a candidate is rebuilt from its configuration.
        :param assets: The approved asset catalog. Unused: a prompt introduces no new asset.
        :returns: The new candidate Agent.
        :raises ValueError: If the harness cannot be serialized or rebuilt.
        """
        return _patched_agent(reference=reference, patch={"system_prompt": self.system_prompt})

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SystemPromptRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'system_prompt'.
        """
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True, kw_only=True)
class ApplyPatchRecipe:
    """
    Apply one approved configuration change from the catalog.

    This is the general axis: because a patch is applied to the harness's serialized form, it reaches any init
    parameter of any component, so reasoning effort, a retriever's `top_k` and a hook's settings all need no recipe
    of their own. The values come from the catalog, so a proposal names a patch and cannot author one.

    :param patch: The name of the approved patch to apply.
    """

    patch: str
    kind: ClassVar[str] = "apply_patch"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Rebuild the Agent from its serialized form with the approved patch applied.

        :param reference: The Agent being optimized. Never modified; a candidate is rebuilt from its configuration.
        :param assets: The catalog holding the approved patch.
        :returns: The new candidate Agent.
        :raises ValueError: If the patch is not approved, the harness does not serialize, or a path does not resolve.
        """
        approved = assets.patch(name=self.patch)
        return _patched_agent(reference=reference, patch=approved.patch)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ApplyPatchRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'patch'.
        """
        return {"kind": self.kind, **asdict(self)}
