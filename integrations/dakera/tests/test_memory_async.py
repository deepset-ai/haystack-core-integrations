# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the async methods of DakeraMemoryStore and its components (respx-mocked)."""

import httpx
import pytest
import respx
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

BASE_URL = "http://localhost:3000"


def _recall_payload():
    return {
        "memories": [
            {
                "memory": {"id": "mem-1", "content": "Alice prefers concise answers", "agent_id": "haystack"},
                "score": 0.91,
            }
        ]
    }


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "test-key")
    return DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False))


@respx.mock
async def test_store_memories_async(store):
    route = respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory": {"id": "m1"}, "embedding_time_ms": 1})
    )
    count = await store.store_memories_async(
        [ChatMessage.from_user("hello"), ChatMessage.from_assistant("")], session_id="s1"
    )
    assert count == 1  # empty message skipped
    assert route.calls.last.request.headers["X-API-Key"] == "test-key"


@respx.mock
async def test_store_memories_async_swallows_errors(store):
    respx.post(f"{BASE_URL}/v1/memory/store").mock(return_value=httpx.Response(500))
    assert await store.store_memories_async([ChatMessage.from_user("x")]) == 0


@respx.mock
async def test_recall_memories_async(store):
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(return_value=httpx.Response(200, json=_recall_payload()))
    messages = await store.recall_memories_async("hello", top_k=3)
    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessage)
    assert messages[0].text == "Alice prefers concise answers"
    assert messages[0].meta["score"] == 0.91


@respx.mock
async def test_recall_memories_async_returns_empty_on_error(store):
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(return_value=httpx.Response(503))
    assert await store.recall_memories_async("hello") == []


@respx.mock
async def test_writer_run_async(store):
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory": {"id": "m1"}, "embedding_time_ms": 1})
    )
    writer = DakeraMemoryWriter(memory_store=store)
    result = await writer.run_async(messages=[ChatMessage.from_user("a"), ChatMessage.from_user("b")])
    assert result["memories_written"] == 2


@respx.mock
async def test_retriever_run_async(store):
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(return_value=httpx.Response(200, json=_recall_payload()))
    retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
    result = await retriever.run_async(query="test")
    assert len(result["memories"]) == 1
    assert isinstance(result["memories"][0], ChatMessage)
