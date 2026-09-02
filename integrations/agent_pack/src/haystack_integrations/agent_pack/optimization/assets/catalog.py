# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog of models and tools a candidate harness may be built from."""

from haystack_integrations.agent_pack.optimization.assets.dataclasses import HarnessPatch, ModelAsset


class ApprovedAssetCatalog:
    """
    The models and configuration changes a candidate harness is allowed to use.

    The schema an optimizer answers against is generated from this catalog, so a proposal naming anything outside
    it fails validation and never reaches a harness. Nothing is re-checked after a candidate is built.
    """

    def __init__(self, *, models: list[ModelAsset], patches: list[HarnessPatch] | None = None) -> None:
        """
        Create an approved asset catalog.

        :param models: Approved model deployments.
        :param patches: Approved configuration changes an experiment may try.
        :raises ValueError: If a model ID or patch name appears twice.
        """
        self.models = {asset.model_id: asset for asset in models}
        self.patches = {declared.name: declared for declared in patches or []}
        if len(self.models) != len(models):
            msg = "Model asset IDs must be unique."
            raise ValueError(msg)
        if len(self.patches) != len(patches or []):
            msg = "Patch names must be unique."
            raise ValueError(msg)

    def model(self, model_id: str) -> ModelAsset:
        """
        Return an approved model or fail closed.

        :param model_id: The model identifier to look up.
        :returns: The approved model asset.
        :raises ValueError: If the model is not in the catalog.
        """
        try:
            return self.models[model_id]
        except KeyError as error:
            msg = f"Model {model_id!r} is not in the approved asset catalog."
            raise ValueError(msg) from error

    def patch(self, name: str) -> HarnessPatch:
        """
        Return an approved patch or fail closed.

        :param name: The patch name to look up.
        :returns: The approved patch.
        :raises ValueError: If the patch is not in the catalog.
        """
        try:
            return self.patches[name]
        except KeyError as error:
            msg = f"Patch {name!r} is not in the approved asset catalog."
            raise ValueError(msg) from error
