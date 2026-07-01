# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DakeraMemoryStore using respx for HTTP mocking."""

import pytest
import respx
import httpx
from haystack.utils import Secret
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore
from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter


BASE_URL = "http://localhost:3300"


@pytest.fixture
def store():
    return DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_token("test-key"))


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
