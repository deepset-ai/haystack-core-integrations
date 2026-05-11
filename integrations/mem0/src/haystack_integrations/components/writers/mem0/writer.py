# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


@component
class Mem0MemoryWriter:
    """
    Writes ChatMessage objects as memories to a Mem0MemoryStore.

    Use this component in a Haystack Pipeline to persist conversation messages.
    Scoping IDs (`user_id`, `run_id`, `agent_id`) are runtime parameters so the
    same pipeline instance can serve multiple users or agents.
    """

    def __init__(self, *, memory_store: Mem0MemoryStore) -> None:
        """
        Initialize the Mem0MemoryWriter.

        :param memory_store: The Mem0MemoryStore instance to write memories to.
        """
        self.memory_store = memory_store

    @component.output_types(memories_written=int)
    def run(
        self,
        messages: list[ChatMessage],
        *,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, int]:
        """
        Write messages as memories to the Mem0 store.

        :param messages: List of ChatMessage objects to store.
        :param user_id: User ID to scope the stored memories.
        :param run_id: Run ID to scope the stored memories.
        :param agent_id: Agent ID to scope the stored memories.
        :returns: Dictionary with key `memories_written` containing the count of stored memory items.
        """
        result = self.memory_store.add_memories(
            messages=messages,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
        )
        count = len(result) if isinstance(result, list) else 0
        return {"memories_written": count}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(self, memory_store=self.memory_store.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryWriter":
        """Deserialize this component from a dictionary."""
        if data.get("init_parameters", {}).get("memory_store"):
            data["init_parameters"]["memory_store"] = Mem0MemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
