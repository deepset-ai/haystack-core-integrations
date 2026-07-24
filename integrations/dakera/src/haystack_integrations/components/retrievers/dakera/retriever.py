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
class DakeraMemoryRetriever:
    """Retrieve relevant memories from a Dakera memory store as ``ChatMessage`` objects.

    Use in Haystack pipelines to inject persistent, decay-weighted context before LLM
    generation. Recalled memories are returned as system-role ``ChatMessage`` objects
    so they can be connected directly to prompt builders and chat generators.

    Args:
        memory_store: The ``DakeraMemoryStore`` to recall from.
        top_k: Number of memories to retrieve. Defaults to 5.

    Example:
        ```python
        from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever

        retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
        result = retriever.run(query="What does Alice prefer?")
        messages = result["memories"]  # list[ChatMessage]
        ```
    """

    def __init__(self, *, memory_store: DakeraMemoryStore, top_k: int = 5) -> None:
        self.memory_store = memory_store
        self.top_k = top_k

    @component.output_types(memories=list[ChatMessage])
    def run(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """Recall memories relevant to the query.

        Args:
            query: Natural language search query.
            agent_id: Optional agent namespace override.
            session_id: Optional session filter.
            tags: Optional tag filter.
            top_k: Overrides the init-time default number of memories to return.

        Returns:
            Dictionary with key ``memories``: a list of system-role ``ChatMessage``
            objects, most relevant first. Dakera fields (``id``, ``score``, ``importance``,
            ...) are available under each message's ``meta``.
        """
        memories = self.memory_store.recall_memories(
            query,
            agent_id=agent_id,
            session_id=session_id,
            tags=tags,
            top_k=top_k if top_k is not None else self.top_k,
        )
        logger.debug("DakeraMemoryRetriever: retrieved {count} memories", count=len(memories))
        return {"memories": memories}

    @component.output_types(memories=list[ChatMessage])
    async def run_async(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """Async version of :meth:`run`."""
        memories = await self.memory_store.recall_memories_async(
            query,
            agent_id=agent_id,
            session_id=session_id,
            tags=tags,
            top_k=top_k if top_k is not None else self.top_k,
        )
        logger.debug("DakeraMemoryRetriever: retrieved {count} memories", count=len(memories))
        return {"memories": memories}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, memory_store=self.memory_store.to_dict(), top_k=self.top_k)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DakeraMemoryRetriever:
        data["init_parameters"]["memory_store"] = DakeraMemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
