# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

logger = logging.getLogger(__name__)


@component
class DakeraMemoryWriter:
    """Write messages to a Dakera memory store.

    Typically placed at the end of a Haystack pipeline to persist
    the conversation exchange for future recall.

    Args:
        memory_store: DakeraMemoryStore instance.

    Example:
        .. code-block:: python

            writer = DakeraMemoryWriter(memory_store=store)
            writer.run(
                messages=["User: what is RAG?", "Assistant: RAG stands for ..."],
                user_id="alice",
            )
    """

    def __init__(self, *, memory_store: DakeraMemoryStore) -> None:
        self.memory_store = memory_store

    @component.output_types(memories_written=int)
    def run(
        self,
        messages: List[str],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Store messages to Dakera.

        Args:
            messages: List of text strings to persist.
            user_id: Optional user identifier.
            agent_id: Optional agent/namespace override.
            session_id: Optional session identifier.

        Returns:
            Dictionary with key ``memories_written``: number of messages stored.
        """
        count = self.memory_store.store_memories(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        logger.debug("DakeraMemoryWriter: stored %d memories", count)
        return {"memories_written": count}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, memory_store=self.memory_store.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DakeraMemoryWriter":
        data["init_parameters"]["memory_store"] = DakeraMemoryStore.from_dict(
            data["init_parameters"]["memory_store"]
        )
        return default_from_dict(cls, data)
