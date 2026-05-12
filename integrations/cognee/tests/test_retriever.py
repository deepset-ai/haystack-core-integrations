# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.cognee import CogneeRetriever
from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


class TestCogneeRetriever:
    def test_init_requires_memory_store(self):
        with pytest.raises(ValueError, match="memory_store must be an instance of CogneeMemoryStore"):
            CogneeRetriever(memory_store="not a store")  # type: ignore[arg-type]

    def test_init_defaults(self):
        store = CogneeMemoryStore()
        retriever = CogneeRetriever(memory_store=store)
        assert retriever._memory_store is store
        assert retriever._top_k is None

    def test_init_with_top_k(self):
        store = CogneeMemoryStore()
        retriever = CogneeRetriever(memory_store=store, top_k=3)
        assert retriever._top_k == 3

    def test_to_from_dict_roundtrip(self):
        store = CogneeMemoryStore(search_type="SUMMARIES", top_k=3, dataset_name="ds")
        retriever = CogneeRetriever(memory_store=store, top_k=7)

        data = retriever.to_dict()
        assert data["type"] == "haystack_integrations.components.retrievers.cognee.memory_retriever.CogneeRetriever"
        assert data["init_parameters"]["top_k"] == 7
        assert (
            data["init_parameters"]["memory_store"]["type"]
            == "haystack_integrations.memory_stores.cognee.memory_store.CogneeMemoryStore"
        )
        assert data["init_parameters"]["memory_store"]["init_parameters"]["dataset_name"] == "ds"

        restored = CogneeRetriever.from_dict(data)
        assert restored._top_k == 7
        assert isinstance(restored._memory_store, CogneeMemoryStore)
        assert restored._memory_store.search_type == "SUMMARIES"
        assert restored._memory_store.dataset_name == "ds"

    def test_run_delegates_to_store(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = [
            ChatMessage.from_system("result one"),
            ChatMessage.from_system("result two"),
        ]

        retriever = CogneeRetriever(memory_store=store)
        out = retriever.run(query="What is Cognee?")

        store.search_memories.assert_called_once_with(query="What is Cognee?", top_k=None, user_id=None)
        assert out["messages"] == store.search_memories.return_value

    def test_run_top_k_override_takes_precedence(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store, top_k=10)
        retriever.run(query="q", top_k=2)

        store.search_memories.assert_called_once_with(query="q", top_k=2, user_id=None)

    def test_run_falls_back_to_init_top_k(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store, top_k=4)
        retriever.run(query="q")

        store.search_memories.assert_called_once_with(query="q", top_k=4, user_id=None)

    def test_run_passes_user_id(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store)
        retriever.run(query="q", user_id="user-abc")

        store.search_memories.assert_called_once_with(query="q", top_k=None, user_id="user-abc")

    def test_run_empty_results(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store)
        assert retriever.run(query="nothing here")["messages"] == []
