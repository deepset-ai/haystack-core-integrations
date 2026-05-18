# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret

from haystack_integrations.memory_stores.mem0.errors import Mem0MemoryStoreError
from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


class TestMem0MemoryStore:
    def test_init_does_not_create_client(self, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        assert store._client is None

    def test_warm_up_creates_client(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        store.warm_up()
        assert store._client is mock_mem0_client

    def test_client_property_triggers_warm_up(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        _ = store.client
        assert store._client is mock_mem0_client

    def test_warm_up_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        store = Mem0MemoryStore()
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
            store.warm_up()

    def test_to_dict(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        result = store.to_dict()
        assert result == {
            "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["MEM0_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
            },
        }

    def test_to_dict_custom_params(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        store = Mem0MemoryStore(api_key=Secret.from_env_var("MY_MEM0_KEY"))
        result = store.to_dict()
        assert result == {
            "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["MY_MEM0_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
            },
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

    def test_from_dict_empty_init_parameters(self):
        data = {
            "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
            "init_parameters": {},
        }
        store = Mem0MemoryStore.from_dict(data)
        assert store.api_key == Secret.from_env_var("MEM0_API_KEY")

    # TODO Check if the return_value is realistic
    def test_add_memories(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": [{"id": "mem-1", "data": {"memory": "User likes Python"}}]}
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        result = store.add_memories(messages=[ChatMessage.from_user("I like Python")], user_id="user-1")

        assert result == [{"memory_id": "mem-1", "memory": "User likes Python"}]
        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "I like Python", "role": "user"}],
            user_id="user-1",
            infer=True,
        )

    def test_add_memories_uses_infer_default(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": []}
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        store.add_memories(messages=[ChatMessage.from_user("test")], user_id="u1")

        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "test", "role": "user"}], infer=True, user_id="u1"
        )

    def test_add_memories_accepts_infer_override(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": []}
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        store.add_memories(messages=[ChatMessage.from_user("test")], user_id="u1", infer=False)

        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "test", "role": "user"}],
            infer=False,
            user_id="u1",
        )

    def test_add_memories_skips_empty_text(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        tc_msg = ChatMessage.from_assistant(text=None, tool_calls=[ToolCall(tool_name="foo", arguments={})])
        result = store.add_memories(messages=[tc_msg], user_id="user-1")
        assert result == []

    def test_add_memories_raises_store_error_on_failure(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.side_effect = Exception("API down")
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        with pytest.raises(Mem0MemoryStoreError, match="Failed to add memories"):
            store.add_memories(messages=[ChatMessage.from_user("test")], user_id="u1")

    def test_search_memories_with_query(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.search.return_value = {"results": [{"memory": "User likes Python", "metadata": None}]}

        store = Mem0MemoryStore()
        results = store.search_memories(query="Python", user_id="user-1")

        assert len(results) == 1
        assert results[0].text == "User likes Python"

    def test_search_memories_without_query_calls_get_all(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.get_all.return_value = {"results": [{"memory": "Memory A", "metadata": {"tag": "work"}}]}

        store = Mem0MemoryStore()
        results = store.search_memories(user_id="user-1")

        mock_mem0_client.get_all.assert_called_once()
        assert results[0].meta == {"tag": "work"}

    def test_search_memories_include_metadata(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        raw = {"memory": "User likes Python", "metadata": None, "id": "mem-1", "score": 0.9}
        mock_mem0_client.search.return_value = {"results": [raw]}

        store = Mem0MemoryStore()
        results = store.search_memories(query="Python", user_id="user-1", include_memory_metadata=True)

        assert "retrieved_memory_metadata" in results[0].meta
        assert results[0].meta["retrieved_memory_metadata"]["id"] == "mem-1"

    def test_search_memories_with_filters_bypasses_id_check(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.get_all.return_value = {"results": []}

        store = Mem0MemoryStore()
        store.search_memories(filters={"field": "user_id", "operator": "==", "value": "user-1"})

        mock_mem0_client.get_all.assert_called_with(filters={"user_id": "user-1"})

    def test_search_memories_raises_store_error_on_failure(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.search.side_effect = Exception("API down")

        store = Mem0MemoryStore()
        with pytest.raises(Mem0MemoryStoreError, match="Failed to search memories"):
            store.search_memories(query="test", user_id="u1")

    def test_get_ids_raises_without_any_id(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        with pytest.raises(ValueError, match="At least one of user_id"):
            store._get_ids()

    def test_get_ids_returns_non_none_only(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        assert store._get_ids(user_id="u1", agent_id="a1", app_id="app1") == {
            "user_id": "u1",
            "agent_id": "a1",
            "app_id": "app1",
        }

    def test_add_memories_passes_app_id(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": []}
        mock_mem0_client.project = Mock()

        store = Mem0MemoryStore()
        store.add_memories(messages=[ChatMessage.from_user("test")], app_id="app1")

        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "test", "role": "user"}],
            infer=True,
            app_id="app1",
        )

    def test_search_memories_with_app_id_filter(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.search.return_value = {"results": []}

        store = Mem0MemoryStore()
        store.search_memories(query="test", app_id="app1")

        mock_mem0_client.search.assert_called_with(query="test", top_k=5, filters={"app_id": "app1"})


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
            store.add_memories(messages=messages, user_id=user_id, infer=False)
            results = store.search_memories(query="Python", user_id=user_id)
            assert any("Python" in (r.text or "") or "Haystack" in (r.text or "") for r in results)
        finally:
            store.client.delete_all(user_id=user_id)
