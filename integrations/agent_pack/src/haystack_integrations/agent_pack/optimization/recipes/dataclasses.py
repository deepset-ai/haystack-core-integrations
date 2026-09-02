# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The closed set of typed, allowlisted transformations for candidate Agent harnesses."""

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.recipes.utils import _clone_agent, _patched_agent


@dataclass(frozen=True, kw_only=True)
class ModelSubstitutionRecipe:
    """
    Replace the coordinator model with an approved model deployment.

    Kept distinct from a patch because a model asset can carry a whole generator configuration, which is what makes
    substituting a model served by a different provider possible, and because it is the axis the deterministic
    proposer enumerates.

    :param model_id: The approved model to switch to.
    """

    model_id: str
    kind: ClassVar[str] = "model_substitution"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Clone the Agent with a generator built by the selected model asset.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :returns: The new candidate Agent.
        """
        generator = assets.model(model_id=self.model_id).build_generator(reference_generator=reference.chat_generator)
        return _clone_agent(reference=reference, chat_generator=generator)

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
        Clone the Agent with the replacement prompt.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist. Unused: a prompt introduces no new asset.
        :returns: The new candidate Agent.
        """
        return _clone_agent(reference=reference, system_prompt=self.system_prompt)

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

        :param reference: The champion harness to transform.
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
