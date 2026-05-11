# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


class TestMem0MemoryStore:
    def test_init(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        assert store.client is mock_mem0_client

    def test_init_explicit_key(self, mock_mem0_client):
        store = Mem0MemoryStore(api_key=Secret.from_token("test-key"))
        assert store.client is mock_mem0_client

    def test_init_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
            Mem0MemoryStore()

    def test_to_dict(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        store = Mem0MemoryStore(api_key=Secret.from_env_var("MY_MEM0_KEY"))
        result = store.to_dict()
        assert result["init_parameters"]["api_key"] == {
            "env_vars": ["MY_MEM0_KEY"],
            "strict": True,
            "type": "env_var",
        }

    def test_from_dict(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        data = {
            "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
            "init_parameters": {
                "api_key": {"env_vars": ["MY_MEM0_KEY"], "strict": True, "type": "env_var"},
            },
        }
        store = Mem0MemoryStore.from_dict(data)
        assert store.api_key == Secret.from_env_var("MY_MEM0_KEY")

    def test_add_memories(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {
            "results": [
                {"id": "mem-1", "data": {"memory": "User likes Python"}},
            ]
        }
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        messages = [ChatMessage.from_user("I like Python")]
        result = store.add_memories(messages=messages, user_id="user-1")

        assert result == [{"memory_id": "mem-1", "memory": "User likes Python"}]
        mock_mem0_client.add.assert_called_once()
        call_kwargs = mock_mem0_client.add.call_args[1]
        assert call_kwargs["user_id"] == "user-1"

    def test_add_memories_skips_empty_text(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": []}
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        tc_msg = ChatMessage.from_assistant(text=None, tool_calls=[ToolCall(tool_name="foo", arguments={})])
        store.add_memories(messages=[tc_msg], user_id="user-1")
        sent = mock_mem0_client.add.call_args[1]["messages"]
        assert sent == []

    def test_search_memories_with_query(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.search.return_value = {"results": [{"memory": "User likes Python", "metadata": None}]}

        store = Mem0MemoryStore()
        results = store.search_memories(query="Python", user_id="user-1")

        assert len(results) == 1
        assert results[0].text == "User likes Python"
        mock_mem0_client.search.assert_called_once()

    def test_search_memories_without_query_calls_get_all(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.get_all.return_value = {"results": [{"memory": "Memory A", "metadata": {"tag": "work"}}]}

        store = Mem0MemoryStore()
        results = store.search_memories(user_id="user-1")

        mock_mem0_client.get_all.assert_called_once()
        assert results[0].meta == {"tag": "work"}

    def test_search_memories_include_metadata(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        raw_memory = {"memory": "User likes Python", "metadata": None, "id": "mem-1", "score": 0.9}
        mock_mem0_client.search.return_value = {"results": [raw_memory]}

        store = Mem0MemoryStore()
        results = store.search_memories(query="Python", user_id="user-1", include_memory_metadata=True)

        assert "retrieved_memory_metadata" in results[0].meta
        assert results[0].meta["retrieved_memory_metadata"]["id"] == "mem-1"

    def test_search_memories_with_filters(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.get_all.return_value = {"results": []}

        store = Mem0MemoryStore()
        filters = {"field": "user_id", "operator": "==", "value": "user-1"}
        store.search_memories(filters=filters)

        call_kwargs = mock_mem0_client.get_all.call_args[1]
        assert call_kwargs["filters"] == {"user_id": "user-1"}

    def test_delete_all_memories(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        store.delete_all_memories(user_id="user-1")
        mock_mem0_client.delete_all.assert_called_once_with(user_id="user-1")

    def test_delete_memory(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        store.delete_memory("mem-42")
        mock_mem0_client.delete.assert_called_once_with(memory_id="mem-42")

    def test_get_ids_raises_without_any_id(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        with pytest.raises(ValueError, match="At least one of user_id"):
            store._get_ids()

    def test_get_ids_returns_non_none_only(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        result = store._get_ids(user_id="u1", run_id=None, agent_id="a1")
        assert result == {"user_id": "u1", "agent_id": "a1"}

    @pytest.mark.parametrize(
        "filters,expected",
        [
            (
                {"field": "score", "operator": ">=", "value": 0.8},
                {"score": {"gte": 0.8}},
            ),
            (
                {"field": "tag", "operator": "==", "value": "python"},
                {"tag": "python"},
            ),
            (
                {"field": "tag", "operator": "!=", "value": "java"},
                {"tag": {"ne": "java"}},
            ),
            (
                {"field": "cat", "operator": "in", "value": ["ml", "nlp"]},
                {"cat": {"in": ["ml", "nlp"]}},
            ),
            (
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "user_id", "operator": "==", "value": "u1"},
                        {"field": "score", "operator": ">", "value": 0.5},
                    ],
                },
                {"AND": [{"user_id": "u1"}, {"score": {"gt": 0.5}}]},
            ),
        ],
    )
    def test_normalize_filters(self, filters, expected):
        assert Mem0MemoryStore.normalize_filters(filters) == expected

    def test_normalize_filters_unsupported_operator(self):
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            Mem0MemoryStore.normalize_filters({"field": "x", "operator": "LIKE", "value": "foo"})


@pytest.mark.skipif(
    not os.environ.get("MEM0_API_KEY"),
    reason="Set MEM0_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestMem0MemoryStoreIntegration:
    def test_add_and_search(self):
        store = Mem0MemoryStore()
        user_id = f"test_{uuid.uuid4().hex}"
        try:
            messages = [ChatMessage.from_user("I love working with Haystack and Python.")]
            store.add_memories(messages=messages, infer=False, user_id=user_id)
            results = store.search_memories(query="Python", user_id=user_id)
            assert any("Python" in (r.text or "") or "Haystack" in (r.text or "") for r in results)
        finally:
            store.delete_all_memories(user_id=user_id)
