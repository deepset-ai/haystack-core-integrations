# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog of models and tools a candidate harness may be built from."""

from haystack_integrations.agent_pack.optimization.assets.dataclasses import HarnessPatch, ModelAsset, ToolAsset


class ApprovedAssetCatalog:
    """
    The models and tools a candidate harness is allowed to use.

    This is the whole compliance boundary, and it is enforced where proposals are parsed: the schema an optimizer
    answers against is generated from this catalog, so a proposal naming anything outside it fails validation and
    never reaches a harness. Nothing needs re-checking after a candidate is built.
    """

    def __init__(
        self,
        *,
        models: list[ModelAsset],
        tools: list[ToolAsset],
        patches: list[HarnessPatch] | None = None,
    ) -> None:
        """
        Create an approved asset catalog.

        :param models: Approved model deployments.
        :param tools: Approved tools.
        :param patches: Approved configuration changes an experiment may try.
        :raises ValueError: If a model ID, tool name or patch name appears twice.
        """
        self.models = {asset.model_id: asset for asset in models}
        self.tools = {asset.name: asset for asset in tools}
        self.patches = {declared.name: declared for declared in patches or []}
        if len(self.models) != len(models):
            msg = "Model asset IDs must be unique."
            raise ValueError(msg)
        if len(self.tools) != len(tools):
            msg = "Tool asset names must be unique."
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

    def tool(self, tool_name: str) -> ToolAsset:
        """
        Return an approved tool or fail closed.

        :param tool_name: The tool name to look up.
        :returns: The approved tool asset.
        :raises ValueError: If the tool is not in the catalog.
        """
        try:
            return self.tools[tool_name]
        except KeyError as error:
            msg = f"Tool {tool_name!r} is not in the approved asset catalog."
            raise ValueError(msg) from error
