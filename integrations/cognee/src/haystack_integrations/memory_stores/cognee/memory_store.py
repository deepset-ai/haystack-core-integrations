# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage

import cognee  # type: ignore[import-untyped]
from cognee.api.v1.search import SearchType  # type: ignore[import-untyped]
from cognee.modules.data.exceptions import DatasetNotFoundError  # type: ignore[import-untyped]
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets  # type: ignore[import-untyped]
from haystack_integrations.components.connectors.cognee._utils import (
    CogneeSearchType,
    _get_cognee_user,
    extract_text,
    run_sync,
)

logger = logging.getLogger(__name__)


class CogneeMemoryStore:
    """
    A memory store backed by Cognee memory.

    Implements the `MemoryStore` protocol from haystack-experimental, allowing
    Cognee to serve as the memory backend for Haystack's experimental Agent.

    Usage:
    ```python
    from haystack_integrations.memory_stores.cognee import CogneeMemoryStore

    store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", top_k=5)
    store.add_memories(messages=[ChatMessage.from_user("Remember: the project deadline is Friday.")])
    results = store.search_memories(query="When is the project deadline?")
    ```
    """

    def __init__(
        self, search_type: CogneeSearchType = "GRAPH_COMPLETION", top_k: int = 5, dataset_name: str = "haystack_memory"
    ):
        """
        Initialize the CogneeMemoryStore.

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
        **kwargs: Any,
    ) -> None:
        """
        Add chat messages to Cognee as memories.

        :param messages: List of ChatMessages to store.
        :param user_id: Optional cognee user UUID to scope memories to a specific user.
            When provided, the data is stored under that user's permissions.
            When ``None``, cognee's default user is used.
        :param kwargs: Additional keyword arguments (unused, accepted for protocol
            compatibility).
        """
        user = run_sync(_get_cognee_user(user_id)) if user_id else None

        added = 0
        for msg in messages:
            text = msg.text
            if not text:
                continue
            run_sync(cognee.add(text, dataset_name=self.dataset_name, user=user))
            added += 1

        if added > 0:
            run_sync(cognee.cognify(datasets=[self.dataset_name], user=user))

        logger.info("Added and cognified {count} messages as memories", count=added)

    def search_memories(
        self,
        *,
        query: str | None = None,
        top_k: int = 5,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Search Cognee's knowledge engine for relevant memories.

        :param query: The search query.
        :param top_k: Maximum number of memories to return.
        :param user_id: Optional cognee user UUID to scope the search to a specific user.
            Search is restricted to the store's ``dataset_name``. If the user owns the
            dataset it is resolved by name; otherwise the store checks whether the user
            has been granted read access (e.g. via shared permissions) and searches by
            dataset UUID.
            When ``None``, cognee's default user is used.
        :param kwargs: Additional keyword arguments (unused, accepted for protocol
            compatibility).
        :returns: List of ChatMessages containing memory content as system messages.
        """
        if not query:
            return []

        user = run_sync(_get_cognee_user(user_id)) if user_id else None
        search_type_enum = SearchType[self.search_type]
        effective_top_k = top_k or self.top_k

        try:
            raw_results = run_sync(
                cognee.search(
                    query_text=query,
                    query_type=search_type_enum,
                    user=user,
                    datasets=[self.dataset_name],
                )
            )
        except DatasetNotFoundError:
            # The user doesn't own a dataset with this name.
            # Fall back to checking shared datasets the user has read access to.
            raw_results = self._search_shared_dataset(query, search_type_enum, user)

        memories: list[ChatMessage] = []
        if not raw_results:
            return memories

        for item in raw_results[:effective_top_k]:
            text = extract_text(item)
            if text:
                memories.append(ChatMessage.from_system(text))

        logger.info("Found {count} memories for query '{query}'", count=len(memories), query=query[:80])
        return memories

    def _search_shared_dataset(self, query: str, search_type_enum: SearchType, user: Any) -> list[Any]:
        """Search for the dataset by name among all datasets the user has read access to."""
        if user is None:
            return []
        all_readable = run_sync(get_all_user_permission_datasets(user, "read"))
        matching = [ds for ds in all_readable if ds.name == self.dataset_name]
        if not matching:
            return []
        return run_sync(
            cognee.search(query_text=query, query_type=search_type_enum, user=user, dataset_ids=[matching[0].id])
        )

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete all memories by pruning Cognee's data and system state.

        :param user_id: Optional cognee user UUID (accepted for protocol compatibility).
            Note: Cognee's prune operations are global and not scoped to a specific user.
        :param kwargs: Additional keyword arguments (unused, accepted for protocol
            compatibility).
        """
        run_sync(cognee.prune.prune_data())
        run_sync(cognee.prune.prune_system(metadata=True))
        logger.info("All Cognee memories pruned")

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a single memory by ID.

        Not supported in V1 — Cognee's SDK does not expose fine-grained deletion.

        :param memory_id: The ID of the memory to delete.
        :raises NotImplementedError: Always, as single-item deletion is not yet supported.
        """
        msg = "CogneeMemoryStore does not support deleting individual memories in V1."
        raise NotImplementedError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            search_type=self.search_type,
            top_k=self.top_k,
            dataset_name=self.dataset_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeMemoryStore":
        """Deserialize a component from a dictionary."""
        return default_from_dict(cls, data)
