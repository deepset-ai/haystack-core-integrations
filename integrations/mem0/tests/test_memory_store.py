# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from typing import ClassVar

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret

from haystack_integrations.memory_stores.mem0.errors import Mem0MemoryStoreError
from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore


def _mem0_memory_result(
    *,
    memory_id: str = "e84c4cdc-6451-41dd-a07a-353eace4a6c0",
    memory: str = "I love working with Haystack and Python.",
    user_id: str = "test_fefdb7b4eebe40549e452f89145ee4f1",
    agent_id: str | None = None,
    app_id: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> dict:
    return {
        "id": memory_id,
        "memory": memory,
        "user_id": user_id,
        "agent_id": agent_id,
        "app_id": app_id,
        "run_id": run_id,
        "score": 0.2416,
        "score_breakdown": {"semantic": 0.6041, "bm25": 0.0, "entity": 0.0},
        "metadata": metadata or {},
        "categories": [],
        "created_at": "2026-05-18T09:52:00.819626+00:00",
        "updated_at": "2026-05-18T09:52:01.067272+00:00",
    }


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

    def test_add_memories(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {
            "results": [
                _mem0_memory_result(
                    memory_id="e84c4cdc-6451-41dd-a07a-353eace4a6c0",
                    memory="I love working with Haystack and Python.",
                    user_id="user-1",
                )
            ]
        }

        store = Mem0MemoryStore()
        result = store.add_memories(messages=[ChatMessage.from_user("I like Python")], user_id="user-1")

        assert result == [
            {
                "memory_id": "e84c4cdc-6451-41dd-a07a-353eace4a6c0",
                "memory": "I love working with Haystack and Python.",
            }
        ]
        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "I like Python", "role": "user"}],
            user_id="user-1",
            infer=True,
        )

    def test_add_memories_uses_infer_default(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": []}

        store = Mem0MemoryStore()
        store.add_memories(messages=[ChatMessage.from_user("test")], user_id="u1")

        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "test", "role": "user"}], infer=True, user_id="u1"
        )

    def test_add_memories_accepts_infer_override(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": []}

        store = Mem0MemoryStore()
        store.add_memories(messages=[ChatMessage.from_user("test")], user_id="u1", infer=False)

        mock_mem0_client.add.assert_called_with(
            messages=[{"content": "test", "role": "user"}],
            infer=False,
            user_id="u1",
        )

    def test_add_memories_skips_empty_text(self, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")

        store = Mem0MemoryStore()
        tc_msg = ChatMessage.from_assistant(text=None, tool_calls=[ToolCall(tool_name="foo", arguments={})])
        result = store.add_memories(messages=[tc_msg], user_id="user-1")
        assert result == []

    def test_add_memories_raises_store_error_on_failure(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.side_effect = Exception("API down")

        store = Mem0MemoryStore()
        with pytest.raises(Mem0MemoryStoreError, match="Failed to add memories"):
            store.add_memories(messages=[ChatMessage.from_user("test")], user_id="u1")

    def test_search_memories_with_query(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.search.return_value = {
            "results": [_mem0_memory_result(memory="User likes Python", user_id="user-1")]
        }

        store = Mem0MemoryStore()
        results = store.search_memories(query="Python", user_id="user-1")

        assert len(results) == 1
        assert results[0].text == "User likes Python"

    def test_search_memories_without_query_calls_get_all(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.get_all.return_value = {
            "results": [_mem0_memory_result(memory="Memory A", user_id="user-1", metadata={"tag": "work"})]
        }

        store = Mem0MemoryStore()
        results = store.search_memories(user_id="user-1")

        mock_mem0_client.get_all.assert_called_once()
        assert results[0].meta["tag"] == "work"
        assert results[0].meta["mem0"]["user_id"] == "user-1"

    def test_search_memories_includes_mem0_metadata(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        raw = _mem0_memory_result(
            memory_id="mem-1",
            memory="User likes Python",
            user_id="user-1",
            metadata={"tag": "work"},
        )
        mock_mem0_client.search.return_value = {"results": [raw]}

        store = Mem0MemoryStore()
        results = store.search_memories(query="Python", user_id="user-1")

        assert results[0].meta["tag"] == "work"
        assert results[0].meta["mem0"] == {
            "memory_id": "mem-1",
            "user_id": "user-1",
            "agent_id": None,
            "app_id": None,
            "run_id": None,
            "score": 0.2416,
            "score_breakdown": {"semantic": 0.6041, "bm25": 0.0, "entity": 0.0},
            "categories": [],
            "created_at": "2026-05-18T09:52:00.819626+00:00",
            "updated_at": "2026-05-18T09:52:01.067272+00:00",
        }

    def test_search_memories_with_filters_only(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.get_all.return_value = {"results": []}

        store = Mem0MemoryStore()
        store.search_memories(filters={"field": "user_id", "operator": "==", "value": "user-1"})

        mock_mem0_client.get_all.assert_called_with(filters={"user_id": "user-1"})

    def test_search_memories_combines_filters_with_ids(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.search.return_value = {"results": []}

        store = Mem0MemoryStore()
        store.search_memories(
            query="Python",
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "tag", "operator": "==", "value": "work"},
                    {"field": "score", "operator": ">=", "value": 0.8},
                ],
            },
            user_id="user-1",
            app_id="app-1",
        )

        mock_mem0_client.search.assert_called_with(
            query="Python",
            top_k=5,
            filters={
                "AND": [
                    {"user_id": "user-1"},
                    {"app_id": "app-1"},
                    {"tag": "work"},
                    {"score": {"gte": 0.8}},
                ]
            },
        )

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
    FILTERABLE_MEMORIES: ClassVar[list[tuple[str, dict[str, str]]]] = [
        ("mem0-filter-work-active", {"category": "work", "status": "active", "topic": "python"}),
        ("mem0-filter-work-archived", {"category": "work", "status": "archived", "topic": "ops"}),
        ("mem0-filter-personal-active", {"category": "personal", "status": "active", "topic": "travel"}),
        ("mem0-filter-finance-draft", {"category": "finance", "status": "draft", "topic": "budget"}),
    ]

    @staticmethod
    def _assert_results_contain_only(results: list[ChatMessage], expected_tokens: set[str]) -> None:
        result_texts = [result.text or "" for result in results]
        assert len(result_texts) == len(expected_tokens)
        assert {token for token in expected_tokens if any(token in text for text in result_texts)} == expected_tokens

    def _seed_filterable_memories(self, store: Mem0MemoryStore, *, user_id: str) -> None:
        for text, metadata in self.FILTERABLE_MEMORIES:
            store.add_memories(
                messages=[ChatMessage.from_user(text)],
                user_id=user_id,
                infer=False,
                metadata=metadata,
            )

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

    def test_search_filters_comparison_operators(self):
        store = Mem0MemoryStore()
        user_id = f"test_{uuid.uuid4().hex}"
        try:
            self._seed_filterable_memories(store, user_id=user_id)

            work_results = store.search_memories(
                user_id=user_id,
                filters={"field": "category", "operator": "==", "value": "work"},
            )
            self._assert_results_contain_only(
                work_results,
                {"mem0-filter-work-active", "mem0-filter-work-archived"},
            )

            not_archived_results = store.search_memories(
                user_id=user_id,
                filters={"field": "status", "operator": "!=", "value": "archived"},
            )
            self._assert_results_contain_only(
                not_archived_results,
                {"mem0-filter-work-active", "mem0-filter-personal-active", "mem0-filter-finance-draft"},
            )
        finally:
            store.client.delete_all(user_id=user_id)

    def test_search_filters_logical_operators(self):
        store = Mem0MemoryStore()
        user_id = f"test_{uuid.uuid4().hex}"
        try:
            self._seed_filterable_memories(store, user_id=user_id)

            and_results = store.search_memories(
                user_id=user_id,
                filters={
                    "operator": "AND",
                    "conditions": [
                        {"field": "category", "operator": "==", "value": "work"},
                        {"field": "status", "operator": "==", "value": "active"},
                    ],
                },
            )
            self._assert_results_contain_only(and_results, {"mem0-filter-work-active"})

            or_results = store.search_memories(
                user_id=user_id,
                filters={
                    "operator": "OR",
                    "conditions": [
                        {"field": "category", "operator": "==", "value": "personal"},
                        {"field": "category", "operator": "==", "value": "finance"},
                    ],
                },
            )
            self._assert_results_contain_only(
                or_results,
                {"mem0-filter-personal-active", "mem0-filter-finance-draft"},
            )
        finally:
            store.client.delete_all(user_id=user_id)

    def test_search_filters_entity_in_operator(self):
        store = Mem0MemoryStore()
        first_user_id = f"test_{uuid.uuid4().hex}"
        second_user_id = f"test_{uuid.uuid4().hex}"
        try:
            store.add_memories(
                messages=[ChatMessage.from_user("mem0-filter-first-user")],
                user_id=first_user_id,
                infer=False,
                metadata={"category": "entity-in"},
            )
            store.add_memories(
                messages=[ChatMessage.from_user("mem0-filter-second-user")],
                user_id=second_user_id,
                infer=False,
                metadata={"category": "entity-in"},
            )

            results = store.search_memories(
                filters={"field": "user_id", "operator": "in", "value": [first_user_id, second_user_id]}
            )

            self._assert_results_contain_only(
                results,
                {"mem0-filter-first-user", "mem0-filter-second-user"},
            )
        finally:
            store.client.delete_all(user_id=first_user_id)
            store.client.delete_all(user_id=second_user_id)
