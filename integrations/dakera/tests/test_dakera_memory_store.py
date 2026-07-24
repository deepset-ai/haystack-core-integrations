# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DakeraMemoryStore using respx for HTTP mocking."""

import httpx
import pytest
import respx
from haystack.utils import Secret

from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

BASE_URL = "http://localhost:3300"


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "test-key")
    return DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False))


@respx.mock
def test_store_memories(store):
    respx.post(f"{BASE_URL}/v1/memories").mock(return_value=httpx.Response(200, json={"id": "mem-1"}))
    count = store.store_memories(["Hello world"], user_id="alice")
    assert count == 1


@respx.mock
def test_search_memories(store):
    respx.post(f"{BASE_URL}/v1/memories/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"id": "mem-1", "content": "Hello world", "score": 0.9}]},
        )
    )
    results = store.search_memories("hello", user_id="alice", top_k=3)
    assert len(results) == 1
    assert results[0]["content"] == "Hello world"
    assert results[0]["score"] == 0.9


@respx.mock
def test_retriever_component(store):
    respx.post(f"{BASE_URL}/v1/memories/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"id": "m1", "content": "Prior context", "score": 0.8}]},
        )
    )
    retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
    result = retriever.run(query="test query")
    assert "memories" in result
    assert len(result["memories"]) == 1


@respx.mock
def test_writer_component(store):
    respx.post(f"{BASE_URL}/v1/memories").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    writer = DakeraMemoryWriter(memory_store=store)
    result = writer.run(messages=["msg1", "msg2"], user_id="bob")
    assert result["memories_written"] == 2


def test_to_dict_round_trip(store):
    d = store.to_dict()
    store2 = DakeraMemoryStore.from_dict(d)
    assert store2.base_url == store.base_url
    assert store2.default_agent_id == store.default_agent_id


@respx.mock
def test_store_memories_swallows_http_errors(store):
    # Exercises the error branch + logger.warning call (regression guard for
    # Haystack's structlog logger, which rejects %-style positional args).
    respx.post(f"{BASE_URL}/v1/memories").mock(return_value=httpx.Response(500, json={"error": "boom"}))
    count = store.store_memories(["one", "two"], user_id="alice")
    assert count == 0


@respx.mock
def test_search_memories_returns_empty_on_error(store):
    respx.post(f"{BASE_URL}/v1/memories/search").mock(return_value=httpx.Response(503))
    assert store.search_memories("hello", user_id="alice") == []


@respx.mock
def test_store_without_api_key_sends_no_auth_header(monkeypatch):
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    store = DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False))
    route = respx.post(f"{BASE_URL}/v1/memories").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    store.store_memories(["hi"])
    assert "Authorization" not in route.calls.last.request.headers


def test_default_base_url_from_env(monkeypatch):
    monkeypatch.setenv("DAKERA_API_URL", "http://example.test:9000/")
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    store = DakeraMemoryStore()
    assert store.base_url == "http://example.test:9000"
    assert store.default_agent_id == "haystack"


def test_writer_to_dict_round_trip(store):
    writer = DakeraMemoryWriter(memory_store=store)
    writer2 = DakeraMemoryWriter.from_dict(writer.to_dict())
    assert writer2.memory_store.base_url == store.base_url


def test_retriever_to_dict_round_trip(store):
    retriever = DakeraMemoryRetriever(memory_store=store, top_k=7)
    retriever2 = DakeraMemoryRetriever.from_dict(retriever.to_dict())
    assert retriever2.top_k == 7
    assert retriever2.memory_store.base_url == store.base_url
