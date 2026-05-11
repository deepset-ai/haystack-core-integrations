# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore

_DEFAULT_DESCRIPTION = (
    "Search long-term memories relevant to a query. "
    "Use this tool whenever you need to recall information from past conversations or stored facts."
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query to find relevant memories."},
        "top_k": {"type": "integer", "description": "Maximum number of memories to return."},
    },
    "required": ["query"],
}


def _retrieve(
    query: str,
    top_k: int,
    store: Mem0MemoryStore,
    user_id: str | None,
    run_id: str | None,
    agent_id: str | None,
) -> str:
    memories: list[ChatMessage] = store.search_memories(
        query=query,
        top_k=top_k,
        user_id=user_id,
        run_id=run_id,
        agent_id=agent_id,
    )
    if not memories:
        return "No memories found."
    return "\n".join(f"- {m.text}" for m in memories if m.text)


class Mem0MemoryRetrieverTool(Tool):
    """
    A tool that searches a Mem0MemoryStore for relevant memories.

    Scoping IDs (``user_id``, ``run_id``, ``agent_id``) are bound at construction time
    and never exposed to the LLM. The LLM only sees ``query`` and ``top_k``.
    """

    def __init__(
        self,
        *,
        memory_store: Mem0MemoryStore,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 5,
        name: str = "retrieve_memories",
        description: str = _DEFAULT_DESCRIPTION,
    ) -> None:
        """
        Initialize the Mem0MemoryRetrieverTool.

        :param memory_store: The Mem0MemoryStore instance to query.
        :param user_id: User ID bound to all searches performed by this tool.
        :param run_id: Run ID bound to all searches performed by this tool.
        :param agent_id: Agent ID bound to all searches performed by this tool.
        :param top_k: Default maximum number of memories to return. The LLM may override this.
        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM.
        """
        self.memory_store = memory_store
        self.user_id = user_id
        self.run_id = run_id
        self.agent_id = agent_id
        self.top_k = top_k
        super().__init__(name=name, description=description, parameters=_PARAMETERS, function=_retrieve)

    def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool, injecting the pre-bound scoping IDs."""
        kwargs.setdefault("store", self.memory_store)
        kwargs.setdefault("user_id", self.user_id)
        kwargs.setdefault("run_id", self.run_id)
        kwargs.setdefault("agent_id", self.agent_id)
        kwargs.setdefault("top_k", self.top_k)
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
                "top_k": self.top_k,
                "name": self.name,
                "description": self.description,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryRetrieverTool":
        """Deserialize this tool from a dictionary."""
        inner = data["data"]
        inner["memory_store"] = Mem0MemoryStore.from_dict(inner["memory_store"])
        return cls(**inner)
