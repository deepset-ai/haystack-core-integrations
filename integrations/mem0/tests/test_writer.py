# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.writers.memory.writer import MemoryWriter
from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMemoryWriter:
    def test_init(self, store):
        writer = MemoryWriter(memory_store=store)
        assert writer.memory_store is store

    def test_run_returns_memories_written(self, store):
        store.add_memories = Mock(
            return_value=[{"memory_id": "m1", "memory": "fact A"}, {"memory_id": "m2", "memory": "fact B"}]
        )
        writer = MemoryWriter(memory_store=store)
        messages = [ChatMessage.from_user("I like Python"), ChatMessage.from_user("I use Haystack")]
        result = writer.run(messages, user_id="u1")
        assert result == {"memories_written": 2}

    def test_run_returns_zero_when_store_returns_none(self, store):
        store.add_memories = Mock(return_value=None)
        writer = MemoryWriter(memory_store=store)
        result = writer.run([ChatMessage.from_user("test")], user_id="u1")
        assert result == {"memories_written": 0}

    def test_run_passes_user_id_as_keyword_only(self, store):
        store.add_memories = Mock(return_value=[])
        writer = MemoryWriter(memory_store=store)
        writer.run([ChatMessage.from_user("test")], user_id="u1")
        kwargs = store.add_memories.call_args[1]
        assert kwargs["user_id"] == "u1"
        assert "run_id" not in kwargs
        assert "agent_id" not in kwargs

    def test_run_with_any_memory_store_duck_typed(self):
        fake_store = Mock()
        fake_store.add_memories.return_value = [{"memory_id": "x"}]
        fake_store.to_dict.return_value = {"type": "fake.Store", "init_parameters": {}}
        writer = MemoryWriter(memory_store=fake_store)
        result = writer.run([ChatMessage.from_user("test")], user_id="u1")
        assert result == {"memories_written": 1}
        assert fake_store.add_memories.call_args[1]["user_id"] == "u1"

    def test_to_dict(self, store):
        writer = MemoryWriter(memory_store=store)
        d = writer.to_dict()
        assert d["type"] == "haystack_integrations.components.writers.memory.writer.MemoryWriter"
        assert "memory_store" in d["init_parameters"]

    def test_from_dict(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.writers.memory.writer.MemoryWriter",
            "init_parameters": {
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {
                        "api_key": {"env_vars": ["MEM0_API_KEY"], "strict": True, "type": "env_var"},
                    },
                },
            },
        }
        writer = MemoryWriter.from_dict(data)
        assert isinstance(writer.memory_store, Mem0MemoryStore)

    def test_serialization_roundtrip(self, store):
        writer = MemoryWriter(memory_store=store)
        d = writer.to_dict()
        assert d["init_parameters"]["memory_store"]["type"].endswith("Mem0MemoryStore")
