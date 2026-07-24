# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DakeraMemoryStore using respx for HTTP mocking."""

import json

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
                "memory": {
                    "id": "mem-1",
                    "content": "Alice prefers concise answers",
                    "agent_id": "haystack",
                    "session_id": "s1",
                    "importance": 0.7,
                    "tags": ["pref"],
                    "metadata": {"source": "chat"},
                    "created_at": 1700000000,
                },
                "score": 0.91,
            }
        ],
        "query_embedding_time_ms": 3,
        "search_time_ms": 5,
    }


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "test-key")
    return DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False))


def test_default_port_is_3000(monkeypatch):
    monkeypatch.delenv("DAKERA_API_URL", raising=False)
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    assert DakeraMemoryStore().base_url == "http://localhost:3000"


@respx.mock
def test_store_memories_posts_to_memory_store_endpoint(store):
    route = respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory": {"id": "mem-1"}, "embedding_time_ms": 2})
    )
    count = store.store_memories([ChatMessage.from_user("Hello world")], session_id="s1")
    assert count == 1
    body = json.loads(route.calls.last.request.read())
    assert body["content"] == "Hello world"
    assert body["agent_id"] == "haystack"
    assert body["session_id"] == "s1"
    # Dakera auth header (not Authorization by default)
    assert route.calls.last.request.headers["X-API-Key"] == "test-key"


@respx.mock
def test_store_skips_empty_messages(store):
    route = respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory": {"id": "m"}, "embedding_time_ms": 1})
    )
    count = store.store_memories([ChatMessage.from_user("real"), ChatMessage.from_assistant("")])
    assert count == 1
    assert route.call_count == 1


@respx.mock
def test_recall_memories_returns_chat_messages(store):
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(return_value=httpx.Response(200, json=_recall_payload()))
    messages = store.recall_memories("hello", top_k=3)
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ChatMessage)
    assert msg.text == "Alice prefers concise answers"
    assert msg.meta["id"] == "mem-1"
    assert msg.meta["score"] == 0.91
    assert msg.meta["source"] == "chat"  # user metadata merged in


@respx.mock
def test_retriever_component(store):
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(return_value=httpx.Response(200, json=_recall_payload()))
    retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
    result = retriever.run(query="test query")
    assert "memories" in result
    assert len(result["memories"]) == 1
    assert isinstance(result["memories"][0], ChatMessage)


@respx.mock
def test_writer_component(store):
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory": {"id": "m1"}, "embedding_time_ms": 1})
    )
    writer = DakeraMemoryWriter(memory_store=store)
    result = writer.run(messages=[ChatMessage.from_user("m1"), ChatMessage.from_assistant("m2")])
    assert result["memories_written"] == 2


@respx.mock
def test_store_swallows_http_errors(store):
    # Exercises the error branch + logger.warning (regression guard: Haystack's structlog
    # logger rejects %-style positional args).
    respx.post(f"{BASE_URL}/v1/memory/store").mock(return_value=httpx.Response(500, json={"error": "boom"}))
    count = store.store_memories([ChatMessage.from_user("one"), ChatMessage.from_user("two")])
    assert count == 0


@respx.mock
def test_recall_returns_empty_on_error(store):
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(return_value=httpx.Response(503))
    assert store.recall_memories("hello") == []


@respx.mock
def test_store_without_api_key_sends_no_auth_header(monkeypatch):
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    store = DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False))
    route = respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=httpx.Response(200, json={"memory": {"id": "m1"}, "embedding_time_ms": 1})
    )
    store.store_memories([ChatMessage.from_user("hi")])
    assert "X-API-Key" not in route.calls.last.request.headers


def test_default_base_url_from_env(monkeypatch):
    monkeypatch.setenv("DAKERA_API_URL", "http://example.test:9000/")
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    store = DakeraMemoryStore()
    assert store.base_url == "http://example.test:9000"
    assert store.default_agent_id == "haystack"


def test_to_dict_round_trip(store):
    store2 = DakeraMemoryStore.from_dict(store.to_dict())
    assert store2.base_url == store.base_url
    assert store2.default_agent_id == store.default_agent_id


def test_writer_to_dict_round_trip(store):
    writer = DakeraMemoryWriter(memory_store=store)
    writer2 = DakeraMemoryWriter.from_dict(writer.to_dict())
    assert writer2.memory_store.base_url == store.base_url


def test_retriever_to_dict_round_trip(store):
    retriever = DakeraMemoryRetriever(memory_store=store, top_k=7)
    retriever2 = DakeraMemoryRetriever.from_dict(retriever.to_dict())
    assert retriever2.top_k == 7
    assert retriever2.memory_store.base_url == store.base_url
