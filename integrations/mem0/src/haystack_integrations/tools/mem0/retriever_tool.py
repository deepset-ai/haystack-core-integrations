# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
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

_DEFAULT_INPUTS_FROM_STATE: dict[str, str] = {"user_id": "user_id"}


class Mem0MemoryRetrieverTool(Tool):
    """
    A tool that searches a Mem0MemoryStore for relevant memories.

    The `user_id` is injected at runtime from Agent State via `inputs_from_state`,
    so a single tool instance can serve many users. The LLM only sees `query` and `top_k`.
    """

    def __init__(
        self,
        *,
        memory_store: Mem0MemoryStore,
        top_k: int = 5,
        name: str = "retrieve_memories",
        description: str = _DEFAULT_DESCRIPTION,
        parameters: dict[str, Any] = _PARAMETERS,
        inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE,
    ) -> None:
        """
        Initialize the Mem0MemoryRetrieverTool.

        :param memory_store: The Mem0MemoryStore instance to query.
        :param top_k: Default maximum number of memories to return. The LLM may override this.
        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM.
        :param parameters: JSON schema for the parameters exposed to the LLM.
        :param inputs_from_state: Mapping of state keys to tool parameter names. Defaults to injecting
            `user_id` from state.
        """
        self.memory_store = memory_store
        self.top_k = top_k
        self._is_warmed_up = False
        super().__init__(
            name=name,
            description=description,
            # We deepcopy to avoid accidental mutations since dicts are mutable and could be shared across instances
            parameters=deepcopy(parameters),
            function=self.retrieve,
            inputs_from_state=dict(inputs_from_state),
        )

    def warm_up(self) -> None:
        """Initialize the Mem0 client. Subsequent calls are no-ops."""
        if self._is_warmed_up:
            return
        self.memory_store.warm_up()
        self._is_warmed_up = True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """
        Retrieve memories relevant to a query.

        :param query: Text query used to search for relevant memories.
        :param top_k: Maximum number of memories to return. Overrides the tool default.
        :param user_id: User ID to scope the search.
        :param run_id: Run ID to scope the search.
        :param agent_id: Agent ID to scope the search.
        :returns: Retrieved memories formatted for the Agent, or a message when no memories were found.
        """
        memories: list[ChatMessage] = self.memory_store.search_memories(
            query=query,
            top_k=top_k if top_k is not None else self.top_k,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
        )
        if not memories:
            return "No memories found."
        return "\n".join(f"- {m.text}" for m in memories if m.text)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "top_k": self.top_k,
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "inputs_from_state": self.inputs_from_state,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryRetrieverTool":
        """Deserialize this tool from a dictionary."""
        inner = data["data"]
        inner["memory_store"] = Mem0MemoryStore.from_dict(inner["memory_store"])
        return cls(**inner)
