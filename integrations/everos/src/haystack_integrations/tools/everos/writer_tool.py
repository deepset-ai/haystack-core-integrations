# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.memory_stores.everos import EverOSMemoryStore

_DEFAULT_DESCRIPTION = (
    "Store durable user-specific information in EverOS long-term memory for future conversations. "
    "Use this for stable facts, preferences, decisions, and important context, not transient requests."
)
_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "The durable information to remember."}},
    "required": ["text"],
}
_DEFAULT_INPUTS_FROM_STATE = {"user_id": "user_id", "session_id": "session_id"}


class EverOSMemoryWriterTool(Tool):
    """A Haystack Agent tool that writes selected information to EverOS user memory."""

    def __init__(
        self,
        *,
        memory_store: EverOSMemoryStore,
        flush_on_write: bool = False,
        name: str = "store_memory",
        description: str = _DEFAULT_DESCRIPTION,
        parameters: dict[str, Any] = _PARAMETERS,
        inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE,
    ) -> None:
        """
        Initialize the EverOS writer tool.

        :param memory_store: EverOS store to write to.
        :param flush_on_write: Force extraction after every tool call instead of waiting for a later boundary.
        :param name: Tool name exposed to the Agent.
        :param description: Tool description exposed to the Agent.
        :param parameters: JSON schema exposed to the Agent.
        :param inputs_from_state: Agent State mapping. Defaults to injecting `user_id` and `session_id`.
        """
        self.memory_store = memory_store
        self.flush_on_write = flush_on_write
        self._is_warmed_up = False
        super().__init__(
            name=name,
            description=description,
            parameters=deepcopy(parameters),
            function=self.store,
            inputs_from_state=dict(inputs_from_state),
        )

    def warm_up(self) -> None:
        """Initialize the EverOS HTTP client. Subsequent calls are no-ops."""
        if self._is_warmed_up:
            return
        self.memory_store.warm_up()
        self._is_warmed_up = True

    def store(
        self,
        text: str,
        *,
        user_id: str,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> str:
        """
        Store one user-memory candidate in EverOS.

        :param text: Durable information to remember.
        :param user_id: User-memory owner, normally injected from Agent State.
        :param session_id: EverOS session buffer, normally injected from Agent State.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :returns: Human-readable add and optional flush status for the Agent.
        """
        result = self.memory_store.add_memories(
            messages=[ChatMessage.from_user(text)],
            session_id=session_id,
            user_id=user_id,
            app_id=app_id,
            project_id=project_id,
            flush=self.flush_on_write,
        )
        message = f"Accepted {result['message_count']} message(s) for EverOS memory (status: {result['status']})."
        if result["flush_status"]:
            message += f" Flush status: {result['flush_status']}."
        return message

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "memory_store": self.memory_store.to_dict(),
                "flush_on_write": self.flush_on_write,
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "inputs_from_state": self.inputs_from_state,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EverOSMemoryWriterTool":
        """Deserialize this tool from a dictionary."""
        inner = data["data"]
        inner["memory_store"] = EverOSMemoryStore.from_dict(inner["memory_store"])
        return cls(**inner)
