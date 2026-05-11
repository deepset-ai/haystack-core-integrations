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

    Use this component in a Haystack Pipeline to persist conversation messages after an
    Agent turn, or wrap it with `ComponentTool` to expose it as an Agent tool.
    """

    def __init__(self, memory_store: Mem0MemoryStore, infer: bool = True) -> None:
        """
        Initialize the Mem0MemoryWriter.

        :param memory_store: The Mem0MemoryStore instance to write memories to.
        :param infer: Default infer setting. If True, Mem0 extracts facts asynchronously.
            If False, messages are stored verbatim and memory IDs are returned immediately.
        """
        self.memory_store = memory_store
        self.infer = infer

    @component.output_types(memories_written=int)
    def run(
        self,
        messages: list[ChatMessage],
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        infer: bool | None = None,
    ) -> dict[str, int]:
        """
        Write messages as memories to Mem0.

        :param messages: List of ChatMessage objects to store.
        :param user_id: User ID to scope the stored memories.
        :param run_id: Run ID to scope the stored memories.
        :param agent_id: Agent ID to scope the stored memories. Required for Mem0 to store assistant messages.
        :param infer: If True, Mem0 extracts facts asynchronously. Overrides the init-time default when set.
        :returns: Dictionary with key `memories_written` containing the count of stored memory items.
        """
        results = self.memory_store.add_memories(
            messages=messages,
            infer=infer if infer is not None else self.infer,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
        )
        return {"memories_written": len(results)}

    @component.output_types(memories_written=int)
    async def run_async(
        self,
        messages: list[ChatMessage],
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        infer: bool | None = None,
    ) -> dict[str, int]:
        """
        Async variant of run(). Delegates to the synchronous implementation.

        :param messages: List of ChatMessage objects to store.
        :param user_id: User ID to scope the stored memories.
        :param run_id: Run ID to scope the stored memories.
        :param agent_id: Agent ID to scope the stored memories. Required for Mem0 to store assistant messages.
        :param infer: If True, Mem0 extracts facts asynchronously. Overrides the init-time default when set.
        :returns: Dictionary with key `memories_written` containing the count of stored memory items.
        """
        return self.run(messages=messages, user_id=user_id, run_id=run_id, agent_id=agent_id, infer=infer)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
            infer=self.infer,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryWriter":
        """Deserialize this component from a dictionary."""
        if "memory_store" in data.get("init_parameters", {}):
            from haystack.core.serialization import default_from_dict as _from_dict  # noqa: PLC0415

            data["init_parameters"]["memory_store"] = _from_dict(
                Mem0MemoryStore, data["init_parameters"]["memory_store"]
            )
        return default_from_dict(cls, data)
