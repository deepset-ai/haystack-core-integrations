# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.connectors.cognee import CogneeMemoryStore


class TestCogneeMemoryStore:
    def test_init_defaults(self):
        store = CogneeMemoryStore()
        assert store.search_type == "GRAPH_COMPLETION"
        assert store.top_k == 5
        assert store.dataset_name == "haystack_memory"

    def test_init_custom(self):
        store = CogneeMemoryStore(search_type="CHUNKS", top_k=10, dataset_name="custom")
        assert store.search_type == "CHUNKS"
        assert store.top_k == 10
        assert store.dataset_name == "custom"

    def test_to_dict(self):
        store = CogneeMemoryStore(search_type="SUMMARIES", top_k=3, dataset_name="mem")
        data = store.to_dict()
        assert data["type"] == "haystack_integrations.components.connectors.cognee.memory_store.CogneeMemoryStore"
        assert data["init_parameters"]["search_type"] == "SUMMARIES"
        assert data["init_parameters"]["top_k"] == 3
        assert data["init_parameters"]["dataset_name"] == "mem"

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.connectors.cognee.memory_store.CogneeMemoryStore",
            "init_parameters": {"search_type": "CHUNKS", "top_k": 8, "dataset_name": "restored"},
        }
        store = CogneeMemoryStore.from_dict(data)
        assert store.search_type == "CHUNKS"
        assert store.top_k == 8
        assert store.dataset_name == "restored"

    @patch("haystack_integrations.components.connectors.cognee.memory_store.cognee")
    def test_add_memories(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        store = CogneeMemoryStore()
        messages = [
            ChatMessage.from_user("Remember: the deadline is Friday."),
            ChatMessage.from_assistant("Got it, I'll remember Friday."),
        ]
        store.add_memories(messages=messages)

        assert mock_cognee.add.await_count == 2
        mock_cognee.cognify.assert_awaited_once()

    @patch("haystack_integrations.components.connectors.cognee.memory_store.cognee")
    def test_search_memories(self, mock_cognee):
        mock_cognee.search = AsyncMock(return_value=["Memory about deadline"])

        store = CogneeMemoryStore()
        results = store.search_memories(query="What is the deadline?")

        assert len(results) == 1
        assert isinstance(results[0], ChatMessage)
        assert results[0].text == "Memory about deadline"

    def test_search_memories_empty_query(self):
        store = CogneeMemoryStore()
        results = store.search_memories(query=None)
        assert results == []

        results = store.search_memories(query="")
        assert results == []

    @patch("haystack_integrations.components.connectors.cognee.memory_store.cognee")
    def test_delete_all_memories(self, mock_cognee):
        mock_cognee.prune = type("Prune", (), {
            "prune_data": AsyncMock(),
            "prune_system": AsyncMock(),
        })()

        store = CogneeMemoryStore()
        store.delete_all_memories()

        mock_cognee.prune.prune_data.assert_awaited_once()
        mock_cognee.prune.prune_system.assert_awaited_once()

    def test_delete_memory_raises(self):
        store = CogneeMemoryStore()
        with pytest.raises(NotImplementedError):
            store.delete_memory("some-id")
