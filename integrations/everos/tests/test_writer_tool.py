# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from haystack_integrations.memory_stores.everos import EverOSMemoryStore
from haystack_integrations.tools.everos import EverOSMemoryWriterTool


def test_writer_tool_schema_and_store_result():
    store = MagicMock(spec=EverOSMemoryStore)
    store.add_memories.return_value = {
        "message_count": 1,
        "status": "accumulated",
        "flush_status": "extracted",
    }
    tool = EverOSMemoryWriterTool(memory_store=store, flush_on_write=True)

    assert tool.parameters["required"] == ["text"]
    assert tool.inputs_from_state == {"user_id": "user_id", "session_id": "session_id"}
    result = tool.store("Alice prefers concise answers.", user_id="alice", session_id="chat-1")

    assert "Accepted 1 message" in result
    assert "Flush status: extracted" in result
    kwargs = store.add_memories.call_args.kwargs
    assert kwargs["messages"][0].text == "Alice prefers concise answers."
    assert kwargs["flush"] is True


def test_writer_tool_round_trip_serialization():
    tool = EverOSMemoryWriterTool(memory_store=EverOSMemoryStore(), flush_on_write=True)
    restored = EverOSMemoryWriterTool.from_dict(tool.to_dict())
    assert isinstance(restored.memory_store, EverOSMemoryStore)
    assert restored.flush_on_write is True
    assert restored.inputs_from_state == {"user_id": "user_id", "session_id": "session_id"}


def test_writer_tool_warm_up_is_idempotent():
    store = MagicMock(spec=EverOSMemoryStore)
    tool = EverOSMemoryWriterTool(memory_store=store)
    tool.warm_up()
    tool.warm_up()
    store.warm_up.assert_called_once()
