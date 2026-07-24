# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging

from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

logger = logging.getLogger(__name__)


@component
class DakeraMemoryRetriever:
    """Retrieve relevant memories from a Dakera memory store.

    Use in Haystack pipelines to inject persistent context before LLM generation.

    Args:
        memory_store: DakeraMemoryStore instance.
        top_k: Number of memories to retrieve. Defaults to 5.

    Example:
        .. code-block:: python

            from haystack import Pipeline
            from haystack.components.generators import OpenAIGenerator
            from haystack_integrations.memory_stores.dakera import DakeraMemoryStore
            from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever

            store = DakeraMemoryStore()
            retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
    """

    def __init__(self, *, memory_store: DakeraMemoryStore, top_k: int = 5) -> None:
        self.memory_store = memory_store
        self.top_k = top_k

    @component.output_types(memories=list)
    def run(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve memories relevant to the query.

        Args:
            query: Natural language search query.
            user_id: Optional user filter.
            agent_id: Optional agent/namespace override.
            session_id: Optional session filter.

        Returns:
            Dictionary with key ``memories``: list of memory dicts (content, score, id).
        """
        memories = self.memory_store.search_memories(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            top_k=self.top_k,
        )
        logger.debug(
            "DakeraMemoryRetriever: retrieved {count} memories for query={query}",
            count=len(memories),
            query=query[:60],
        )
        return {"memories": memories}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, memory_store=self.memory_store.to_dict(), top_k=self.top_k)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DakeraMemoryRetriever:
        data["init_parameters"]["memory_store"] = DakeraMemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
