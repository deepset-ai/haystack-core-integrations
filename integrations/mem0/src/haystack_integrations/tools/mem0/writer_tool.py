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


def _store(
    text: str,
    store: Mem0MemoryStore,
    user_id: str | None,
    run_id: str | None,
    agent_id: str | None,
) -> str:
    result = store.add_memories(
        messages=[ChatMessage.from_user(text)],
        user_id=user_id,
        run_id=run_id,
        agent_id=agent_id,
    )
    count = len(result) if isinstance(result, list) else 0
    return f"Stored {count} memory item(s)."


class Mem0MemoryWriterTool(Tool):
    """
    A tool that writes a memory to a Mem0MemoryStore.

    Scoping IDs (``user_id``, ``run_id``, ``agent_id``) are bound at construction time
    and never exposed to the LLM. The LLM only sees ``text``.
    """

    def __init__(
        self,
        *,
        memory_store: Mem0MemoryStore,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        name: str = "store_memory",
        description: str = _DEFAULT_DESCRIPTION,
    ) -> None:
        """
        Initialize the Mem0MemoryWriterTool.

        :param memory_store: The Mem0MemoryStore instance to write to.
        :param user_id: User ID bound to all writes performed by this tool.
        :param run_id: Run ID bound to all writes performed by this tool.
        :param agent_id: Agent ID bound to all writes performed by this tool.
        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM.
        """
        self.memory_store = memory_store
        self.user_id = user_id
        self.run_id = run_id
        self.agent_id = agent_id
        super().__init__(name=name, description=description, parameters=_PARAMETERS, function=_store)

    def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool, injecting the pre-bound scoping IDs."""
        kwargs.setdefault("store", self.memory_store)
        kwargs.setdefault("user_id", self.user_id)
        kwargs.setdefault("run_id", self.run_id)
        kwargs.setdefault("agent_id", self.agent_id)
        return super().invoke(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "user_id": self.user_id,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
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
