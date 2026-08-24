# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol for sources of captured Agent runs."""

from typing import Protocol

from haystack_integrations.agent_pack.optimization.tracing.dataclasses import TraceArtifact, TraceSelection


class TraceSource(Protocol):
    """Source of Haystack trace artifacts."""

    def list(self, selection: TraceSelection | None = None) -> list[TraceArtifact]:
        """
        Return artifacts matching the selection.

        :param selection: Which artifacts to return. Every artifact when omitted.
        :returns: The matching artifacts.
        """
        ...
