# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore

_DEFAULT_DESCRIPTION = (
    "Store a piece of information as a long-term memory. "
    "Use this tool to persist important facts, preferences, or context for future conversations."
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The information to store as a memory."},
    },
    "required": ["text"],
}


class Mem0MemoryWriterTool(Tool):
    """
    A tool that writes a memory to a Mem0MemoryStore.

    All scoping IDs (`user_id`, `run_id`, `agent_id`) are injected at runtime
    from Agent State via `inputs_from_state`, so a single tool instance can serve
    many users and sessions. The LLM only sees `text`.
    """

    def __init__(
        self,
        *,
        memory_store: Mem0MemoryStore,
        name: str = "store_memory",
        description: str = _DEFAULT_DESCRIPTION,
    ) -> None:
        """
        Initialize the Mem0MemoryWriterTool.

        :param memory_store: The Mem0MemoryStore instance to write to.
        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM.
        """
        self.memory_store = memory_store
        super().__init__(
            name=name,
            description=description,
            parameters=_PARAMETERS,
            function=self._store,
            inputs_from_state={"user_id": "user_id", "run_id": "run_id", "agent_id": "agent_id"},
        )

    def warm_up(self) -> None:
        """Initialize the Mem0 client by warming up the underlying memory store."""
        self.memory_store.warm_up()

    def _store(
        self,
        text: str,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        result = self.memory_store.add_memories(
            messages=[ChatMessage.from_user(text)],
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
        )
        count = len(result) if isinstance(result, list) else 0
        return f"Stored {count} memory item(s)."

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "name": self.name,
                "description": self.description,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryWriterTool":
        """Deserialize this tool from a dictionary."""
        inner = data["data"]
        inner["memory_store"] = Mem0MemoryStore.from_dict(inner["memory_store"])
        return cls(**inner)
