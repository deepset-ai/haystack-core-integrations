# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Literal

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.memory_stores.everos import EverOSMemoryStore

_DEFAULT_DESCRIPTION = (
    "Search EverOS long-term memory for relevant context from earlier conversations. "
    "Use this before answering when remembered user facts, preferences, or prior decisions may help."
)
_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "A focused query for relevant long-term memories."},
        "top_k": {"type": "integer", "description": "Maximum results per EverOS memory kind."},
    },
    "required": ["query"],
}
_DEFAULT_INPUTS_FROM_STATE = {"user_id": "user_id"}


class EverOSMemoryRetrieverTool(Tool):
    """A Haystack Agent tool that searches EverOS user or agent memory."""

    def __init__(
        self,
        *,
        memory_store: EverOSMemoryStore,
        top_k: int = 5,
        method: Literal["keyword", "vector", "hybrid", "agentic"] = "hybrid",
        include_profile: bool = False,
        name: str = "retrieve_memories",
        description: str = _DEFAULT_DESCRIPTION,
        parameters: dict[str, Any] = _PARAMETERS,
        inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE,
    ) -> None:
        """
        Initialize the EverOS retriever tool.

        :param memory_store: EverOS store to search.
        :param top_k: Default result limit per memory kind.
        :param method: EverOS retrieval method.
        :param include_profile: Include user profile data in user-memory searches.
        :param name: Tool name exposed to the Agent.
        :param description: Tool description exposed to the Agent.
        :param parameters: JSON schema exposed to the Agent.
        :param inputs_from_state: Agent State mapping. Defaults to injecting `user_id`.
        """
        self.memory_store = memory_store
        self.top_k = top_k
        self.method = method
        self.include_profile = include_profile
        self._is_warmed_up = False
        super().__init__(
            name=name,
            description=description,
            parameters=deepcopy(parameters),
            function=self.retrieve,
            inputs_from_state=dict(inputs_from_state),
        )

    def warm_up(self) -> None:
        """Initialize the EverOS HTTP client. Subsequent calls are no-ops."""
        if self._is_warmed_up:
            return
        self.memory_store.warm_up()
        self._is_warmed_up = True

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
    ) -> str:
        """
        Search EverOS and format the memories for the Agent.

        :param query: Focused retrieval query.
        :param top_k: Runtime result-limit override.
        :param user_id: User-memory owner, injected from Agent State by default.
        :param agent_id: Agent-memory owner. Use a custom state mapping instead of `user_id` to search this track.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :returns: Bulleted memory context or a no-results message.
        """
        memories = self.memory_store.search_memories(
            query=query,
            top_k=top_k if top_k is not None else self.top_k,
            method=self.method,
            include_profile=self.include_profile,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            project_id=project_id,
        )
        if not memories:
            return "No memories found."
        return "\n\n".join(f"- {message.text}" for message in memories if message.text)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "top_k": self.top_k,
                "method": self.method,
                "include_profile": self.include_profile,
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "inputs_from_state": self.inputs_from_state,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EverOSMemoryRetrieverTool":
        """Deserialize this tool from a dictionary."""
        inner = data["data"]
        inner["memory_store"] = EverOSMemoryStore.from_dict(inner["memory_store"])
        return cls(**inner)
