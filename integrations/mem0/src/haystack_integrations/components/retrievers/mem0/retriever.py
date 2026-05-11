# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


@component
class Mem0MemoryRetriever:
    """
    Retrieves memories from a Mem0MemoryStore as a list of ChatMessage objects.

    Use this component in a Haystack Pipeline to fetch relevant memories before passing
    context to a language model, or wrap it with `ComponentTool` to expose it as an Agent tool.
    """

    def __init__(self, memory_store: Mem0MemoryStore, top_k: int = 5) -> None:
        """
        Initialize the Mem0MemoryRetriever.

        :param memory_store: The Mem0MemoryStore instance to retrieve memories from.
        :param top_k: Default maximum number of memories to return per query.
        """
        self.memory_store = memory_store
        self.top_k = top_k

    @component.output_types(memories=list[ChatMessage])
    def run(
        self,
        query: str,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        include_memory_metadata: bool = False,
    ) -> dict[str, list[ChatMessage]]:
        """
        Retrieve memories matching the query from Mem0.

        :param query: Text query used to search for relevant memories.
        :param user_id: User ID to scope the search.
        :param run_id: Run ID to scope the search.
        :param agent_id: Agent ID to scope the search.
        :param filters: Haystack-style filters to apply. When provided, ID parameters are ignored.
        :param top_k: Maximum number of memories to return. Overrides the init-time default.
        :param include_memory_metadata: If True, each ChatMessage's meta will include a
            `retrieved_memory_metadata` key with the raw Mem0 memory object.
        :returns: Dictionary with key `memories` containing a list of ChatMessage objects.
        """
        memories = self.memory_store.search_memories(
            query=query,
            filters=filters,
            top_k=top_k if top_k is not None else self.top_k,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
            include_memory_metadata=include_memory_metadata,
        )
        return {"memories": memories}

    @component.output_types(memories=list[ChatMessage])
    async def run_async(
        self,
        query: str,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        include_memory_metadata: bool = False,
    ) -> dict[str, list[ChatMessage]]:
        """
        Async variant of run(). Delegates to the synchronous implementation.

        :param query: Text query used to search for relevant memories.
        :param user_id: User ID to scope the search.
        :param run_id: Run ID to scope the search.
        :param agent_id: Agent ID to scope the search.
        :param filters: Haystack-style filters to apply. When provided, ID parameters are ignored.
        :param top_k: Maximum number of memories to return. Overrides the init-time default.
        :param include_memory_metadata: If True, each ChatMessage's meta will include a
            `retrieved_memory_metadata` key with the raw Mem0 memory object.
        :returns: Dictionary with key `memories` containing a list of ChatMessage objects.
        """
        return self.run(
            query=query,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
            filters=filters,
            top_k=top_k,
            include_memory_metadata=include_memory_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
            top_k=self.top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryRetriever":
        """Deserialize this component from a dictionary."""
        if "memory_store" in data.get("init_parameters", {}):
            from haystack.core.serialization import default_from_dict as _from_dict  # noqa: PLC0415

            data["init_parameters"]["memory_store"] = _from_dict(
                Mem0MemoryStore, data["init_parameters"]["memory_store"]
            )
        return default_from_dict(cls, data)
