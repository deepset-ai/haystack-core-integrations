# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog of models and tools a candidate harness is allowed to use, and validation against it."""

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.assets.dataclasses import AssetValidation, ModelAsset, ToolAsset

__all__ = ["ApprovedAssetCatalog", "AssetValidation", "ModelAsset", "ToolAsset"]
