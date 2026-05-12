# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


@component
class CogneeRetriever:
    """
    Retrieves memories from a `CogneeMemoryStore` as `ChatMessage` instances.

    Configuration (`search_type`, `top_k`, `dataset_name`, `session_id`) lives on
    the store; this retriever is a thin pipeline adapter over `search_memories`.
    """

    def __init__(self, *, memory_store: CogneeMemoryStore, top_k: int | None = None):
        """
        Initialize the retriever.

        :param memory_store: Backing `CogneeMemoryStore` to query.
        :param top_k: Default max results; falls back to the store's `top_k` when `None`.
            Overridable per-call via `run`.
        """
        if not isinstance(memory_store, CogneeMemoryStore):
            msg = "memory_store must be an instance of CogneeMemoryStore"
            raise ValueError(msg)
        self._memory_store = memory_store
        self._top_k = top_k

    @component.output_types(messages=list[ChatMessage])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Search the attached store and return matching memories as ChatMessages.

        :param query: Natural-language query.
        :param top_k: Per-call override; falls back to init `top_k`, then the store's default.
        :param user_id: Cognee user UUID; scopes the search to that user.
        """
        effective_top_k = top_k if top_k is not None else self._top_k
        messages = self._memory_store.search_memories(query=query, top_k=effective_top_k, user_id=user_id)
        return {"messages": messages}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(self, memory_store=self._memory_store.to_dict(), top_k=self._top_k)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeRetriever":
        """Deserialize a component from a dictionary."""
        data["init_parameters"]["memory_store"] = CogneeMemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
