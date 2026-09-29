# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.everos import EverOSMemoryStore
from haystack_integrations.tools.everos import EverOSMemoryRetrieverTool


def test_retriever_tool_formats_memories():
    store = MagicMock(spec=EverOSMemoryStore)
    store.search_memories.return_value = [
        ChatMessage.from_system("Alice prefers concise answers."),
        ChatMessage.from_system("Alice works on developer tooling."),
    ]
    tool = EverOSMemoryRetrieverTool(memory_store=store, top_k=6, method="hybrid", include_profile=True)

    result = tool.retrieve("What does Alice prefer?", user_id="alice", top_k=2)

    assert "Alice prefers concise answers" in result
    assert "Alice works on developer tooling" in result
    store.search_memories.assert_called_once_with(
        query="What does Alice prefer?",
        top_k=2,
        method="hybrid",
        include_profile=True,
        user_id="alice",
        agent_id=None,
        app_id="default",
        project_id="default",
    )


def test_retriever_tool_returns_no_memories_message():
    store = MagicMock(spec=EverOSMemoryStore)
    store.search_memories.return_value = []
    tool = EverOSMemoryRetrieverTool(memory_store=store)
    assert tool.retrieve("query", user_id="alice") == "No memories found."


def test_retriever_tool_round_trip_serialization():
    tool = EverOSMemoryRetrieverTool(memory_store=EverOSMemoryStore(), top_k=8, method="keyword", include_profile=True)
    restored = EverOSMemoryRetrieverTool.from_dict(tool.to_dict())
    assert isinstance(restored.memory_store, EverOSMemoryStore)
    assert restored.top_k == 8
    assert restored.method == "keyword"
    assert restored.include_profile is True


def test_retriever_tool_warm_up_is_idempotent():
    store = MagicMock(spec=EverOSMemoryStore)
    tool = EverOSMemoryRetrieverTool(memory_store=store)
    tool.warm_up()
    tool.warm_up()
    store.warm_up.assert_called_once()
