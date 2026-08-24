# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol for candidate harness transformations."""

from typing import Any, ClassVar, Protocol

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog


class CandidateRecipe(Protocol):
    """A constrained, serializable transformation from one Agent to another."""

    kind: ClassVar[str]

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Create a candidate without mutating the reference Agent.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :returns: The new candidate Agent.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the recipe into a dictionary for hashing and campaign journaling.

        :returns: A dictionary whose `kind` key identifies the transformation.
        """
        ...
