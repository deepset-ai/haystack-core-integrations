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

    def test_to_dict(self):
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

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.retrievers.cognee.memory_retriever.CogneeRetriever",
            "init_parameters": {
                "top_k": 5,
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.cognee.memory_store.CogneeMemoryStore",
                    "init_parameters": {
                        "search_type": "CHUNKS",
                        "top_k": 8,
                        "dataset_name": "restored",
                    },
                },
            },
        }
        retriever = CogneeRetriever.from_dict(data)
        assert retriever._top_k == 5
        assert isinstance(retriever._memory_store, CogneeMemoryStore)
        assert retriever._memory_store.search_type == "CHUNKS"
        assert retriever._memory_store.dataset_name == "restored"

    def test_run_delegates_to_store(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = [
            ChatMessage.from_system("result one"),
            ChatMessage.from_system("result two"),
        ]

        retriever = CogneeRetriever(memory_store=store)
        result = retriever.run(query="What is Cognee?")

        store.search_memories.assert_called_once_with(query="What is Cognee?", user_id=None)
        docs = result["documents"]
        assert len(docs) == 2
        assert docs[0].content == "result one"
        assert docs[0].meta["source"] == "cognee"

    def test_run_forwards_top_k_override(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store, top_k=10)
        retriever.run(query="q", top_k=2)

        store.search_memories.assert_called_once_with(query="q", user_id=None, top_k=2)

    def test_run_uses_retriever_default_top_k(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store, top_k=4)
        retriever.run(query="q")

        store.search_memories.assert_called_once_with(query="q", user_id=None, top_k=4)

    def test_run_forwards_user_id(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store)
        retriever.run(query="q", user_id="550e8400-e29b-41d4-a716-446655440000")

        store.search_memories.assert_called_once_with(
            query="q",
            user_id="550e8400-e29b-41d4-a716-446655440000",
        )

    def test_run_empty_results(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = []

        retriever = CogneeRetriever(memory_store=store)
        result = retriever.run(query="nonexistent query")

        assert result["documents"] == []

    def test_run_skips_messages_with_empty_text(self):
        store = MagicMock(spec=CogneeMemoryStore)
        store.search_memories.return_value = [
            ChatMessage.from_system("valid"),
            ChatMessage.from_system(""),
        ]

        retriever = CogneeRetriever(memory_store=store)
        result = retriever.run(query="q")

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "valid"
