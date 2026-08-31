# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.writers.everos import EverOSMemoryWriter
from haystack_integrations.memory_stores.everos import EverOSMemoryStore


def test_writer_forwards_runtime_scope_and_reports_server_semantics():
    store = MagicMock(spec=EverOSMemoryStore)
    store.add_memories.return_value = {
        "message_count": 2,
        "status": "accumulated",
        "flush_status": "extracted",
    }
    writer = EverOSMemoryWriter(memory_store=store, flush_on_write=True)
    messages = [ChatMessage.from_user("Remember this"), ChatMessage.from_assistant("Understood")]

    result = writer.run(
        messages,
        session_id="session-1",
        user_id="alice",
        agent_id="agent-1",
        app_id="haystack",
        project_id="demo",
        defer_extraction=True,
    )

    assert result == {"messages_written": 2, "status": "accumulated", "flush_status": "extracted"}
    store.add_memories.assert_called_once_with(
        messages=messages,
        session_id="session-1",
        user_id="alice",
        agent_id="agent-1",
        app_id="haystack",
        project_id="demo",
        defer_extraction=True,
        flush=True,
    )


def test_writer_round_trip_serialization():
    writer = EverOSMemoryWriter(memory_store=EverOSMemoryStore(), flush_on_write=True)
    restored = EverOSMemoryWriter.from_dict(writer.to_dict())
    assert isinstance(restored.memory_store, EverOSMemoryStore)
    assert restored.flush_on_write is True
