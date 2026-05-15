# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


def _graph(text: str) -> SimpleNamespace:
    return SimpleNamespace(source="graph", text=text)


def _session(answer: str, question: str = "") -> SimpleNamespace:
    return SimpleNamespace(source="session", answer=answer, question=question)


class TestCogneeMemoryStore:
    def test_init_defaults(self):
        store = CogneeMemoryStore()
        assert store.search_type == "GRAPH_COMPLETION"
        assert store.top_k == 5
        assert store.dataset_name == "haystack_memory"
        assert store.session_id is None

    def test_init_custom(self):
        store = CogneeMemoryStore(search_type="CHUNKS", top_k=10, dataset_name="custom", session_id="s1")
        assert store.search_type == "CHUNKS"
        assert store.top_k == 10
        assert store.dataset_name == "custom"
        assert store.session_id == "s1"

    def test_init_default_self_improvement(self):
        """Mirrors cognee.remember's default of self_improvement=True."""
        assert CogneeMemoryStore().self_improvement is True

    def test_to_from_dict_roundtrip(self):
        store = CogneeMemoryStore(
            search_type="SUMMARIES",
            top_k=3,
            dataset_name="mem",
            session_id="abc",
            self_improvement=False,
        )
        data = store.to_dict()
        assert data["type"] == "haystack_integrations.memory_stores.cognee.memory_store.CogneeMemoryStore"
        assert data["init_parameters"] == {
            "search_type": "SUMMARIES",
            "top_k": 3,
            "dataset_name": "mem",
            "session_id": "abc",
            "self_improvement": False,
        }
        restored = CogneeMemoryStore.from_dict(data)
        assert restored.search_type == "SUMMARIES"
        assert restored.top_k == 3
        assert restored.dataset_name == "mem"
        assert restored.session_id == "abc"
        assert restored.self_improvement is False

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_batches_permanent_tier(self, mock_cognee):
        mock_cognee.remember = AsyncMock()
        store = CogneeMemoryStore(dataset_name="my_ds")

        store.add_memories(
            messages=[
                ChatMessage.from_user("First fact"),
                ChatMessage.from_assistant("Second fact"),
            ]
        )

        mock_cognee.remember.assert_awaited_once_with(
            ["First fact", "Second fact"], dataset_name="my_ds", user=None, self_improvement=True
        )

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_session_tier_writes_one_per_message(self, mock_cognee):
        mock_cognee.remember = AsyncMock()
        store = CogneeMemoryStore(dataset_name="ds", session_id="sess1")

        store.add_memories(messages=[ChatMessage.from_user("a"), ChatMessage.from_user("b")])

        assert mock_cognee.remember.await_count == 2
        first_call = mock_cognee.remember.await_args_list[0]
        assert first_call.args == ("a",)
        assert first_call.kwargs == {
            "dataset_name": "ds",
            "session_id": "sess1",
            "user": None,
            "self_improvement": True,
        }

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_self_improvement_false_forwarded(self, mock_cognee):
        """`self_improvement=False` flows through to cognee.remember (both tiers)."""
        mock_cognee.remember = AsyncMock()

        perm = CogneeMemoryStore(dataset_name="ds", self_improvement=False)
        perm.add_memories(messages=[ChatMessage.from_user("a")])
        assert mock_cognee.remember.await_args.kwargs["self_improvement"] is False

        mock_cognee.remember.reset_mock()
        sess = CogneeMemoryStore(dataset_name="ds", session_id="s", self_improvement=False)
        sess.add_memories(messages=[ChatMessage.from_user("a")])
        assert mock_cognee.remember.await_args.kwargs["self_improvement"] is False

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_session_id_override_writes_to_override(self, mock_cognee):
        """`session_id` kwarg routes a permanent-tier store's write into a session."""
        mock_cognee.remember = AsyncMock()
        store = CogneeMemoryStore(dataset_name="ds")  # no session_id on the store

        store.add_memories(messages=[ChatMessage.from_user("a")], session_id="call_sess")

        mock_cognee.remember.assert_awaited_once_with(
            "a", dataset_name="ds", session_id="call_sess", user=None, self_improvement=True
        )

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_session_id_override_beats_store_session(self, mock_cognee):
        """Per-call `session_id` wins over `self.session_id`."""
        mock_cognee.remember = AsyncMock()
        store = CogneeMemoryStore(dataset_name="ds", session_id="store_sess")

        store.add_memories(messages=[ChatMessage.from_user("a")], session_id="call_sess")

        assert mock_cognee.remember.await_args.kwargs["session_id"] == "call_sess"

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_skips_when_no_text(self, mock_cognee):
        mock_cognee.remember = AsyncMock()
        store = CogneeMemoryStore()

        store.add_memories(messages=[])
        store.add_memories(messages=[ChatMessage.from_user("")])

        mock_cognee.remember.assert_not_called()

    @patch("haystack_integrations.memory_stores.cognee.memory_store._resolve_user", new_callable=AsyncMock)
    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_resolves_user(self, mock_cognee, mock_resolve):
        mock_cognee.remember = AsyncMock()
        sentinel_user = object()
        mock_resolve.return_value = sentinel_user
        uid = str(uuid4())

        store = CogneeMemoryStore(dataset_name="ds")
        store.add_memories(messages=[ChatMessage.from_user("hi")], user_id=uid)

        mock_resolve.assert_awaited_once_with(uid)
        mock_cognee.remember.assert_awaited_once_with(
            ["hi"], dataset_name="ds", user=sentinel_user, self_improvement=True
        )

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_search_memories_passes_recall_args(self, mock_cognee):
        mock_cognee.SearchType = {"GRAPH_COMPLETION": "GRAPH_COMPLETION_ENUM"}
        mock_cognee.recall = AsyncMock(return_value=[])

        store = CogneeMemoryStore(dataset_name="ds", top_k=7, session_id="s")
        store.search_memories(query="hello")

        mock_cognee.recall.assert_awaited_once_with(
            "hello",
            query_type="GRAPH_COMPLETION_ENUM",
            datasets=["ds"],
            top_k=7,
            session_id="s",
            user=None,
        )

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_search_memories_wraps_results_per_source(self, mock_cognee):
        mock_cognee.SearchType = {"GRAPH_COMPLETION": MagicMock()}
        mock_cognee.recall = AsyncMock(
            return_value=[
                _graph("graph hit"),
                _session("session answer", "question text"),
                SimpleNamespace(source="graph_context", content="ctx blob"),
                SimpleNamespace(source="trace", memory_context="trace blob"),
            ]
        )

        store = CogneeMemoryStore()
        out = store.search_memories(query="q")

        assert [m.text for m in out] == [
            "graph hit",
            "session answer",
            "ctx blob",
            "trace blob",
        ]
        assert all(isinstance(m, ChatMessage) for m in out)

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_search_memories_top_k_override(self, mock_cognee):
        mock_cognee.SearchType = {"GRAPH_COMPLETION": MagicMock()}
        mock_cognee.recall = AsyncMock(return_value=[])

        store = CogneeMemoryStore(top_k=5)
        store.search_memories(query="q", top_k=2)

        assert mock_cognee.recall.await_args.kwargs["top_k"] == 2

    def test_search_memories_empty_query(self):
        store = CogneeMemoryStore()
        assert store.search_memories(query=None) == []
        assert store.search_memories(query="") == []

    @patch("haystack_integrations.memory_stores.cognee.memory_store._resolve_user", new_callable=AsyncMock)
    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_search_memories_resolves_user(self, mock_cognee, mock_resolve):
        mock_cognee.SearchType = {"GRAPH_COMPLETION": MagicMock()}
        mock_cognee.recall = AsyncMock(return_value=[])
        sentinel_user = object()
        mock_resolve.return_value = sentinel_user
        uid = str(uuid4())

        store = CogneeMemoryStore()
        store.search_memories(query="q", user_id=uid)

        mock_resolve.assert_awaited_once_with(uid)
        assert mock_cognee.recall.await_args.kwargs["user"] is sentinel_user

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_improve_uses_store_session(self, mock_cognee):
        mock_cognee.improve = AsyncMock()

        store = CogneeMemoryStore(dataset_name="ds", session_id="s1")
        store.improve()

        mock_cognee.improve.assert_awaited_once_with(dataset="ds", session_ids=["s1"], user=None)

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_improve_overrides_session(self, mock_cognee):
        mock_cognee.improve = AsyncMock()

        store = CogneeMemoryStore(dataset_name="ds")
        store.improve(session_id="explicit")

        mock_cognee.improve.assert_awaited_once_with(dataset="ds", session_ids=["explicit"], user=None)

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_improve_no_session_runs_graph_enrichment(self, mock_cognee):
        mock_cognee.improve = AsyncMock()

        store = CogneeMemoryStore(dataset_name="ds")
        store.improve()

        mock_cognee.improve.assert_awaited_once_with(dataset="ds", session_ids=None, user=None)

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_delete_all_memories_calls_forget_with_dataset(self, mock_cognee):
        mock_cognee.forget = AsyncMock()

        store = CogneeMemoryStore(dataset_name="ds")
        store.delete_all_memories()

        mock_cognee.forget.assert_awaited_once_with(dataset="ds", user=None)
