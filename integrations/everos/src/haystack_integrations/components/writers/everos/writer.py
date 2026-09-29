# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.everos import EverOSMemoryStore


@component
class EverOSMemoryWriter:
    """
    Write Haystack chat messages into an EverOS session buffer.

    EverOS derives durable episodes, facts, profiles, agent cases, and skills from message streams. The component
    therefore reports accepted message count and extraction status instead of claiming one memory per message.
    """

    def __init__(self, *, memory_store: EverOSMemoryStore, flush_on_write: bool = False) -> None:
        """
        Initialize the writer.

        :param memory_store: EverOS store to write to.
        :param flush_on_write: Force extraction after each run instead of waiting for a later semantic boundary.
        """
        self.memory_store = memory_store
        self.flush_on_write = flush_on_write

    @component.output_types(messages_written=int, status=str, flush_status=str | None)
    def run(
        self,
        messages: list[ChatMessage],
        *,
        session_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
        defer_extraction: bool = False,
    ) -> dict[str, Any]:
        """
        Add messages to EverOS.

        :param messages: Conversation messages to add.
        :param session_id: Conversation or run identifier.
        :param user_id: Owner ID for user messages.
        :param agent_id: Sender ID for assistant and tool messages.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :param defer_extraction: Buffer without boundary detection until a later flush.
        :returns: Accepted message count, add status, and optional flush status.
        """
        result = self.memory_store.add_memories(
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            project_id=project_id,
            defer_extraction=defer_extraction,
            flush=self.flush_on_write,
        )
        return {
            "messages_written": result["message_count"],
            "status": result["status"],
            "flush_status": result["flush_status"],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
            flush_on_write=self.flush_on_write,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EverOSMemoryWriter":
        """Deserialize this component from a dictionary."""
        if data.get("init_parameters", {}).get("memory_store"):
            data["init_parameters"]["memory_store"] = EverOSMemoryStore.from_dict(
                data["init_parameters"]["memory_store"]
            )
        return default_from_dict(cls, data)
