# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

logger = logging.getLogger(__name__)


@component
class DakeraMemoryWriter:
    """Write ``ChatMessage`` objects as memories to a Dakera memory store.

    Typically placed at the end of a Haystack pipeline to persist the conversation
    exchange for future recall.

    Args:
        memory_store: The ``DakeraMemoryStore`` to write to.

    Example:
        ```python
        from haystack.dataclasses import ChatMessage
        from haystack_integrations.components.writers.dakera import DakeraMemoryWriter

        writer = DakeraMemoryWriter(memory_store=store)
        writer.run(messages=[ChatMessage.from_user("Alice prefers concise Python examples.")])
        ```
    """

    def __init__(self, *, memory_store: DakeraMemoryStore) -> None:
        self.memory_store = memory_store

    @component.output_types(memories_written=int)
    def run(
        self,
        messages: list[ChatMessage],
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, int]:
        """Store the text of each message to Dakera.

        Args:
            messages: Chat messages to persist. Messages with no text are skipped.
            agent_id: Optional agent namespace override.
            session_id: Optional session identifier.
            tags: Optional tags attached to each stored memory.

        Returns:
            Dictionary with key ``memories_written``: number of messages stored.
        """
        count = self.memory_store.store_memories(
            messages,
            agent_id=agent_id,
            session_id=session_id,
            tags=tags,
        )
        logger.debug("DakeraMemoryWriter: stored {count} memories", count=count)
        return {"memories_written": count}

    @component.output_types(memories_written=int)
    async def run_async(
        self,
        messages: list[ChatMessage],
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, int]:
        """Async version of :meth:`run`."""
        count = await self.memory_store.store_memories_async(
            messages,
            agent_id=agent_id,
            session_id=session_id,
            tags=tags,
        )
        logger.debug("DakeraMemoryWriter: stored {count} memories", count=count)
        return {"memories_written": count}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, memory_store=self.memory_store.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DakeraMemoryWriter:
        data["init_parameters"]["memory_store"] = DakeraMemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
