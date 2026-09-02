# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog of models and configuration changes a candidate harness may be built from."""

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.assets.dataclasses import HarnessPatch, ModelAsset

__all__ = ["ApprovedAssetCatalog", "HarnessPatch", "ModelAsset"]
