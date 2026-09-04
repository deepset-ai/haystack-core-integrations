# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.everos import EverOSMemoryRetriever
from haystack_integrations.memory_stores.everos import EverOSMemoryStore


def test_retriever_forwards_options():
    store = MagicMock(spec=EverOSMemoryStore)
    expected = [ChatMessage.from_system("Remembered context")]
    store.search_memories.return_value = expected
    retriever = EverOSMemoryRetriever(memory_store=store, top_k=7, method="keyword", include_profile=True)

    result = retriever.run(
        "What does Alice prefer?",
        user_id="alice",
        session_id="session-1",
        top_k=3,
        radius=0.5,
        min_score=0.6,
    )

    assert result == {"memories": expected}
    store.search_memories.assert_called_once_with(
        query="What does Alice prefer?",
        user_id="alice",
        agent_id=None,
        app_id="default",
        project_id="default",
        session_id="session-1",
        filters=None,
        method="keyword",
        top_k=3,
        radius=0.5,
        min_score=0.6,
        include_profile=True,
        enable_llm_rerank=False,
        include_unprocessed=False,
    )


def test_retriever_round_trip_serialization():
    retriever = EverOSMemoryRetriever(memory_store=EverOSMemoryStore(), top_k=9, method="agentic", include_profile=True)
    restored = EverOSMemoryRetriever.from_dict(retriever.to_dict())
    assert isinstance(restored.memory_store, EverOSMemoryStore)
    assert restored.top_k == 9
    assert restored.method == "agentic"
    assert restored.include_profile is True
