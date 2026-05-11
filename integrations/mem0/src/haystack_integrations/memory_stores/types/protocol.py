# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

from haystack.dataclasses import ChatMessage


@runtime_checkable
class MemoryStore(Protocol):
    """
    Minimal protocol for memory stores that persist and retrieve ChatMessage-based memories.

    Only defines what Haystack components require: serialization, adding memories, and
    searching memories. Store-specific features (additional scoping IDs, filters, deletion)
    belong on the concrete implementation.
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
        :returns: List of stored memory items if the store reports them, or ``None``.
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
