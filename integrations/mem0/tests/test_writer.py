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
        result = writer.run([ChatMessage.from_user("I like Python")], user_id="u1")
        assert result == {"memories_written": 2}

    def test_run_returns_zero_when_store_returns_none(self, store):
        store.add_memories = Mock(return_value=None)
        writer = Mem0MemoryWriter(memory_store=store)
        result = writer.run([ChatMessage.from_user("test")], user_id="u1")
        assert result == {"memories_written": 0}

    def test_run_passes_all_ids(self, store):
        store.add_memories = Mock(return_value=[])
        writer = Mem0MemoryWriter(memory_store=store, infer=False)
        messages = [ChatMessage.from_user("test")]
        writer.run(messages, user_id="u1", run_id="r2", agent_id="a3", app_id="app1")
        store.add_memories.assert_called_once_with(
            messages=messages,
            user_id="u1",
            run_id="r2",
            agent_id="a3",
            app_id="app1",
            infer=False,
        )

    def test_to_dict(self, store):
        writer = Mem0MemoryWriter(memory_store=store, infer=False)
        data = writer.to_dict()
        assert data == {
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
        assert isinstance(writer.memory_store, Mem0MemoryStore)
        assert writer.infer is False
