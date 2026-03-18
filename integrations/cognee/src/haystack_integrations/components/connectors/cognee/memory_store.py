# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage

import cognee  # type: ignore[import-untyped]
from cognee.api.v1.search import SearchType  # type: ignore[import-untyped]

from ._utils import run_sync

logger = logging.getLogger(__name__)


class CogneeMemoryStore:
    """
    A memory store backed by Cognee memory.

    Implements the `MemoryStore` protocol from haystack-experimental, allowing
    Cognee to serve as the memory backend for Haystack's experimental Agent.

    Usage:
    ```python
    from haystack_integrations.components.connectors.cognee import CogneeMemoryStore

    store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", top_k=5)
    store.add_memories(messages=[ChatMessage.from_user("Remember: the project deadline is Friday.")])
    results = store.search_memories(query="When is the project deadline?")
    ```
    """

    def __init__(self, search_type: str = "GRAPH_COMPLETION", top_k: int = 5, dataset_name: str = "haystack_memory"):
        """
        :param search_type: Cognee search type for memory retrieval.
        :param top_k: Default number of results for memory search.
        :param dataset_name: Cognee dataset name for storing memories.
        """
        self.search_type = search_type
        self.top_k = top_k
        self.dataset_name = dataset_name

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add chat messages to Cognee as memories and cognify them.

        :param messages: List of ChatMessages to store.
        :param user_id: Optional user identifier (reserved for future use).
        :param agent_id: Optional agent identifier (reserved for future use).
        :param run_id: Optional run identifier (reserved for future use).
        """
        for msg in messages:
            text = msg.text
            if not text:
                continue
            run_sync(cognee.add(text, dataset_name=self.dataset_name))

        run_sync(cognee.cognify())
        logger.info("Added and cognified {count} messages as memories", count=len(messages))

    def search_memories(
        self,
        *,
        query: str | None = None,
        top_k: int = 5,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Search Cognee's knowledge engine for relevant memories.

        :param query: The search query.
        :param top_k: Maximum number of memories to return.
        :param user_id: Optional user identifier (reserved for future use).
        :param agent_id: Optional agent identifier (reserved for future use).
        :param run_id: Optional run identifier (reserved for future use).
        :returns: List of ChatMessages containing memory content as system messages.
        """
        if not query:
            return []

        search_type_enum = SearchType[self.search_type]
        effective_top_k = top_k or self.top_k

        raw_results = run_sync(cognee.search(query_text=query, query_type=search_type_enum))

        memories: list[ChatMessage] = []
        if not raw_results:
            return memories

        for item in raw_results[:effective_top_k]:
            text = _extract_memory_text(item)
            if text:
                memories.append(ChatMessage.from_system(text))

        logger.info("Found {count} memories for query '{query}'", count=len(memories), query=query[:80])
        return memories

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete all memories by pruning Cognee's data and system state.

        :param user_id: Optional user identifier (reserved for future use).
        :param agent_id: Optional agent identifier (reserved for future use).
        :param run_id: Optional run identifier (reserved for future use).
        """
        run_sync(cognee.prune.prune_data())
        run_sync(cognee.prune.prune_system(metadata=True))
        logger.info("All Cognee memories pruned")

    def delete_memory(self, memory_id: str, **kwargs: Any) -> None:
        """
        Delete a single memory by ID.

        Not supported in V1 — Cognee's SDK does not expose fine-grained deletion.

        :param memory_id: The ID of the memory to delete.
        :raises NotImplementedError: Always, as single-item deletion is not yet supported.
        """
        msg = "CogneeMemoryStore does not support deleting individual memories in V1."
        raise NotImplementedError(msg)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            search_type=self.search_type,
            top_k=self.top_k,
            dataset_name=self.dataset_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeMemoryStore":
        return default_from_dict(cls, data)


def _extract_memory_text(item: Any) -> str:
    """Best-effort text extraction from a Cognee search result item."""
    if isinstance(item, str):
        return item

    for attr in ("content", "text", "description", "name"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            if val and isinstance(val, str):
                return val

    if isinstance(item, dict):
        for key in ("content", "text", "description", "name"):
            if key in item and isinstance(item[key], str):
                return item[key]

    return str(item)
