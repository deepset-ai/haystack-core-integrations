# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.writers.mem0.writer import Mem0MemoryWriter
from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMem0MemoryWriter:
    def test_init(self, store):
        writer = Mem0MemoryWriter(memory_store=store, infer=False)
        assert writer.memory_store is store
        assert writer.infer is False

    def test_run_returns_memories_written(self, store):
        store.add_memories = Mock(
            return_value=[{"memory_id": "m1", "memory": "fact A"}, {"memory_id": "m2", "memory": "fact B"}]
        )
        writer = Mem0MemoryWriter(memory_store=store)
        messages = [ChatMessage.from_user("I like Python"), ChatMessage.from_user("I use Haystack")]
        result = writer.run(messages=messages, user_id="u1")
        assert result == {"memories_written": 2}

    def test_run_passes_infer_override(self, store):
        store.add_memories = Mock(return_value=[])
        writer = Mem0MemoryWriter(memory_store=store, infer=True)
        writer.run(messages=[ChatMessage.from_user("test")], user_id="u1", infer=False)
        assert store.add_memories.call_args[1]["infer"] is False

    def test_run_uses_default_infer(self, store):
        store.add_memories = Mock(return_value=[])
        writer = Mem0MemoryWriter(memory_store=store, infer=False)
        writer.run(messages=[ChatMessage.from_user("test")], user_id="u1")
        assert store.add_memories.call_args[1]["infer"] is False

    def test_run_passes_ids(self, store):
        store.add_memories = Mock(return_value=[])
        writer = Mem0MemoryWriter(memory_store=store)
        writer.run(messages=[ChatMessage.from_user("test")], user_id="u1", run_id="r2", agent_id="a3")
        kwargs = store.add_memories.call_args[1]
        assert kwargs["user_id"] == "u1"
        assert kwargs["run_id"] == "r2"
        assert kwargs["agent_id"] == "a3"

    @pytest.mark.asyncio
    async def test_run_async_delegates_to_run(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "x", "memory": "fact"}])
        writer = Mem0MemoryWriter(memory_store=store)
        result = await writer.run_async(messages=[ChatMessage.from_user("test")], user_id="u1")
        assert result == {"memories_written": 1}

    def test_to_dict(self, store):
        writer = Mem0MemoryWriter(memory_store=store, infer=False)
        d = writer.to_dict()
        assert d["type"] == "haystack_integrations.components.writers.mem0.writer.Mem0MemoryWriter"
        assert d["init_parameters"]["infer"] is False
        assert "memory_store" in d["init_parameters"]

    def test_from_dict(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.writers.mem0.writer.Mem0MemoryWriter",
            "init_parameters": {
                "infer": False,
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {
                        "api_key": {"env_vars": ["MEM0_API_KEY"], "strict": True, "type": "env_var"},
                    },
                },
            },
        }
        writer = Mem0MemoryWriter.from_dict(data)
        assert writer.infer is False
        assert isinstance(writer.memory_store, Mem0MemoryStore)

    def test_serialization_roundtrip(self, store):
        writer = Mem0MemoryWriter(memory_store=store, infer=True)
        d = writer.to_dict()
        assert d["init_parameters"]["infer"] is True
        assert d["init_parameters"]["memory_store"]["type"].endswith("Mem0MemoryStore")
