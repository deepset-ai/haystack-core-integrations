# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.mem0.retriever import Mem0MemoryRetriever
from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMem0MemoryRetriever:
    def test_init(self, store):
        retriever = Mem0MemoryRetriever(memory_store=store, top_k=10)
        assert retriever.memory_store is store
        assert retriever.top_k == 10

    def test_run_returns_memories_key(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("Memory A")])
        retriever = Mem0MemoryRetriever(memory_store=store)
        result = retriever.run("test", user_id="u1")
        assert "memories" in result
        assert result["memories"][0].text == "Memory A"
        store.search_memories.assert_called_once_with(
            query="test",
            filters=None,
            top_k=5,
            user_id="u1",
            run_id=None,
            agent_id=None,
            app_id=None,
            include_memory_metadata=False,
        )

    def test_run_top_k_override(self, store):
        store.search_memories = Mock(return_value=[])
        retriever = Mem0MemoryRetriever(memory_store=store, top_k=3)
        retriever.run("test", user_id="u1", top_k=7)
        store.search_memories.assert_called_once_with(
            query="test",
            filters=None,
            top_k=7,
            user_id="u1",
            run_id=None,
            agent_id=None,
            app_id=None,
            include_memory_metadata=False,
        )

    def test_run_uses_default_top_k_when_not_overridden(self, store):
        store.search_memories = Mock(return_value=[])
        retriever = Mem0MemoryRetriever(memory_store=store, top_k=3)
        retriever.run("test", user_id="u1")
        store.search_memories.assert_called_once_with(
            query="test",
            filters=None,
            top_k=3,
            user_id="u1",
            run_id=None,
            agent_id=None,
            app_id=None,
            include_memory_metadata=False,
        )

    def test_run_keyword_only_after_query(self, store):
        store.search_memories = Mock(return_value=[])
        retriever = Mem0MemoryRetriever(memory_store=store)
        retriever.run("test", user_id="u1", run_id="r1", agent_id="a1", app_id="app1")
        store.search_memories.assert_called_once_with(
            query="test",
            filters=None,
            top_k=5,
            user_id="u1",
            run_id="r1",
            agent_id="a1",
            app_id="app1",
            include_memory_metadata=False,
        )

    def test_to_dict(self, store):
        retriever = Mem0MemoryRetriever(memory_store=store, top_k=7)
        d = retriever.to_dict()
        assert d["type"] == "haystack_integrations.components.retrievers.mem0.retriever.Mem0MemoryRetriever"
        assert d["init_parameters"]["top_k"] == 7
        assert "memory_store" in d["init_parameters"]

    def test_from_dict(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.retrievers.mem0.retriever.Mem0MemoryRetriever",
            "init_parameters": {
                "top_k": 8,
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {
                        "api_key": {"env_vars": ["MEM0_API_KEY"], "strict": True, "type": "env_var"},
                    },
                },
            },
        }
        retriever = Mem0MemoryRetriever.from_dict(data)
        assert retriever.top_k == 8
        assert isinstance(retriever.memory_store, Mem0MemoryStore)

    def test_serialization_roundtrip(self, store):
        retriever = Mem0MemoryRetriever(memory_store=store, top_k=4)
        d = retriever.to_dict()
        assert d["init_parameters"]["top_k"] == 4
        assert d["init_parameters"]["memory_store"]["type"].endswith("Mem0MemoryStore")
