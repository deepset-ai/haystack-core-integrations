# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


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
        assert data["type"] == "haystack_integrations.memory_stores.cognee.memory_store.CogneeMemoryStore"
        assert data["init_parameters"]["search_type"] == "SUMMARIES"
        assert data["init_parameters"]["top_k"] == 3
        assert data["init_parameters"]["dataset_name"] == "mem"

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.memory_stores.cognee.memory_store.CogneeMemoryStore",
            "init_parameters": {"search_type": "CHUNKS", "top_k": 8, "dataset_name": "restored"},
        }
        store = CogneeMemoryStore.from_dict(data)
        assert store.search_type == "CHUNKS"
        assert store.top_k == 8
        assert store.dataset_name == "restored"

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
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

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_cognify_uses_dataset(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        store = CogneeMemoryStore(dataset_name="my_ds")
        messages = [ChatMessage.from_user("Remember this.")]
        store.add_memories(messages=messages)

        mock_cognee.cognify.assert_awaited_once()
        cognify_kwargs = mock_cognee.cognify.call_args[1]
        assert cognify_kwargs["datasets"] == ["my_ds"]

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
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

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_delete_all_memories(self, mock_cognee):
        mock_cognee.prune = type(
            "Prune",
            (),
            {
                "prune_data": AsyncMock(),
                "prune_system": AsyncMock(),
            },
        )()

        store = CogneeMemoryStore()
        store.delete_all_memories()

        mock_cognee.prune.prune_data.assert_awaited_once()
        mock_cognee.prune.prune_system.assert_awaited_once()

    @patch("haystack_integrations.memory_stores.cognee.memory_store._get_cognee_user", new_callable=AsyncMock)
    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_with_user_id(self, mock_cognee, mock_get_user):
        mock_user = MagicMock()
        mock_get_user.return_value = mock_user
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        store = CogneeMemoryStore()
        messages = [ChatMessage.from_user("Remember this.")]
        store.add_memories(messages=messages, user_id="550e8400-e29b-41d4-a716-446655440000")

        mock_get_user.assert_awaited_once_with("550e8400-e29b-41d4-a716-446655440000")
        mock_cognee.add.assert_awaited_once()
        add_kwargs = mock_cognee.add.call_args[1]
        assert add_kwargs["user"] is mock_user
        cognify_kwargs = mock_cognee.cognify.call_args[1]
        assert cognify_kwargs["user"] is mock_user

    @patch("haystack_integrations.memory_stores.cognee.memory_store._get_cognee_user", new_callable=AsyncMock)
    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_search_memories_with_user_id(self, mock_cognee, mock_get_user):
        mock_user = MagicMock()
        mock_get_user.return_value = mock_user
        mock_cognee.search = AsyncMock(return_value=["result"])

        store = CogneeMemoryStore()
        results = store.search_memories(query="test", user_id="550e8400-e29b-41d4-a716-446655440000")

        mock_get_user.assert_awaited_once_with("550e8400-e29b-41d4-a716-446655440000")
        search_kwargs = mock_cognee.search.call_args[1]
        assert search_kwargs["user"] is mock_user
        assert len(results) == 1

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_add_memories_without_user_id(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        store = CogneeMemoryStore()
        messages = [ChatMessage.from_user("Remember this.")]
        store.add_memories(messages=messages)

        add_kwargs = mock_cognee.add.call_args[1]
        assert add_kwargs["user"] is None

    @patch("haystack_integrations.memory_stores.cognee.memory_store.cognee")
    def test_search_memories_without_user_id(self, mock_cognee):
        mock_cognee.search = AsyncMock(return_value=["result"])

        store = CogneeMemoryStore()
        store.search_memories(query="test")

        search_kwargs = mock_cognee.search.call_args[1]
        assert search_kwargs["user"] is None

    def test_delete_memory_raises(self):
        store = CogneeMemoryStore()
        with pytest.raises(NotImplementedError):
            store.delete_memory("some-id")
