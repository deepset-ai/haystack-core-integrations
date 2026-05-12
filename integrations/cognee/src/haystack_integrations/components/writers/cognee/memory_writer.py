# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


@component
class CogneeWriter:
    """
    Persists `ChatMessage`s into a `CogneeMemoryStore`.

    Use without `session_id` to write to the permanent graph; pass `session_id` to
    target cognee's session cache for that writer's writes. The writer's
    `session_id` overrides the store's own `session_id` per call, so one store can
    back multiple writers writing to different tiers.
    """

    def __init__(
        self,
        *,
        memory_store: CogneeMemoryStore,
        session_id: str | None = None,
    ):
        """
        Initialize the writer.

        :param memory_store: Backing `CogneeMemoryStore` to write into.
        :param session_id: When set, writes go to cognee's session cache under this
            id; overrides the store's `session_id` for this writer. When `None`,
            falls back to the store's tier.
        """
        if not isinstance(memory_store, CogneeMemoryStore):
            msg = "memory_store must be an instance of CogneeMemoryStore"
            raise ValueError(msg)
        self._memory_store = memory_store
        self._session_id = session_id

    @component.output_types(messages_written=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        user_id: str | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Store `messages` in Cognee memory and pass them through unchanged.

        :param messages: Messages to persist.
        :param user_id: Cognee user UUID; scopes the write to that user.
        """
        self._memory_store.add_memories(messages=messages, user_id=user_id, session_id=self._session_id)
        return {"messages_written": messages}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(self, memory_store=self._memory_store.to_dict(), session_id=self._session_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeWriter":
        """Deserialize a component from a dictionary."""
        data["init_parameters"]["memory_store"] = CogneeMemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
