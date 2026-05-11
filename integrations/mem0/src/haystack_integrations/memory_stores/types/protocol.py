# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

from haystack.dataclasses import ChatMessage


@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for memory stores that persist and retrieve ChatMessage-based memories.

    This protocol defines the minimal interface shared by all memory store implementations.
    Only ``user_id`` is included as a scoping parameter because it is the common denominator
    across known implementations. Store-specific scoping (e.g. ``run_id``, ``agent_id`` for
    Mem0) and store-specific query features (e.g. ``filters``) are extensions that belong on
    the concrete implementation, not on this protocol.

    The ``MemoryWriter`` component is typed against this protocol and therefore only exposes
    ``user_id`` at runtime. To use additional scoping IDs, type your component or pipeline
    against the concrete store class (e.g. ``Mem0MemoryStore``) directly.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize this memory store to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryStore":
        """Deserialize a memory store from a dictionary."""
        ...

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        user_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Add ChatMessage memories to the store.

        :param messages: List of ChatMessage objects to store as memories.
        :param user_id: User ID to scope the stored memories.
        :returns: List of stored memory items (e.g. with IDs and text) if the store
            supports reporting them, or ``None`` if a count is not available.
        """
        ...

    def search_memories(
        self,
        *,
        query: str | None = None,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[ChatMessage]:
        """
        Search for memories in the store.

        :param query: Text query to search. If omitted, all memories in scope are returned.
        :param top_k: Maximum number of results to return.
        :param user_id: User ID to scope the search.
        :returns: List of ChatMessage objects containing the retrieved memories.
        """
        ...

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
    ) -> None:
        """
        Delete all memories in the given scope.

        :param user_id: User ID whose memories to delete.
        """
        ...

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a single memory by ID.

        :param memory_id: The ID of the memory to delete.
        :raises NotImplementedError: If the store does not support single-item deletion.
        """
        ...
