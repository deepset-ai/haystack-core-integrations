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
    "Store a piece of information as a long-term memory. "
    "Use this tool to persist important facts, preferences, or context for future conversations."
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The information to store as a memory."},
        "infer": {
            "type": "boolean",
            "description": "If true, Mem0 extracts memories from the text. If false, Mem0 stores the text as-is.",
            "default": False,
        },
    },
    "required": ["text"],
}

_DEFAULT_INPUTS_FROM_STATE: dict[str, str] = {"user_id": "user_id"}


class Mem0MemoryWriterTool(Tool):
    """
    A tool that writes a memory to a Mem0MemoryStore.

    The `user_id` is injected at runtime from Agent State via `inputs_from_state`,
    so a single tool instance can serve many users. The LLM only sees `text` and `infer`.
    """

    def __init__(
        self,
        *,
        memory_store: Mem0MemoryStore,
        name: str = "store_memory",
        description: str = _DEFAULT_DESCRIPTION,
        parameters: dict[str, Any] = _PARAMETERS,
        inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE,
    ) -> None:
        """
        Initialize the Mem0MemoryWriterTool.

        :param memory_store: The Mem0MemoryStore instance to write to.
        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM.
        :param parameters: JSON schema for the parameters exposed to the LLM.
        :param inputs_from_state: Mapping of state keys to tool parameter names. Defaults to injecting
            `user_id` from state.
        """
        self.memory_store = memory_store
        self._is_warmed_up = False
        super().__init__(
            name=name,
            description=description,
            # We deepcopy to avoid accidental mutations since dicts are mutable and could be shared across instances
            parameters=deepcopy(parameters),
            function=self.store,
            inputs_from_state=dict(inputs_from_state),
        )

    def warm_up(self) -> None:
        """Initialize the Mem0 client. Subsequent calls are no-ops."""
        if self._is_warmed_up:
            return
        self.memory_store.warm_up()
        self._is_warmed_up = True

    def store(
        self,
        text: str,
        infer: bool = False,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
    ) -> str:
        """
        Store text as a memory.

        :param text: The information to store as a memory.
        :param user_id: User ID to scope the stored memory.
        :param run_id: Run ID to scope the stored memory.
        :param agent_id: Agent ID to scope the stored memory.
        :param app_id: App ID to scope the stored memory.
        :param infer: If True, Mem0 extracts memories from the text. If False, Mem0 stores the text as-is.
        :returns: A string indicating how many memory items were stored.
        """
        result = self.memory_store.add_memories(
            messages=[ChatMessage.from_user(text)],
            infer=infer,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
            app_id=app_id,
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
                "parameters": self.parameters,
                "inputs_from_state": self.inputs_from_state,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryWriterTool":
        """Deserialize this tool from a dictionary."""
        inner = data["data"]
        inner["memory_store"] = Mem0MemoryStore.from_dict(inner["memory_store"])
        inner.pop("infer", None)
        return cls(**inner)
