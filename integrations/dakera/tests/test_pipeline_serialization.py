# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end serialization + execution of both integrations inside a Haystack Pipeline."""

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import respx
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.retrievers.dakera import DakeraEmbeddingRetriever, DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter
from haystack_integrations.document_stores.dakera import DakeraDocumentStore
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

BASE_URL = "http://localhost:3000"


@respx.mock
def test_memory_pipeline_dumps_loads_and_runs(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "dk-test")
    store = DakeraMemoryStore(base_url=BASE_URL, api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False))
    pipe = Pipeline()
    pipe.add_component("writer", DakeraMemoryWriter(memory_store=store))
    pipe.add_component("retriever", DakeraMemoryRetriever(memory_store=store, top_k=3))

    # YAML round-trip: a pipeline built from env-var secrets serializes and deserializes cleanly.
    restored = Pipeline.loads(pipe.dumps())

    respx.post(f"{BASE_URL}/v1/memory/store").mock(return_value=httpx.Response(200, json={"memory": {"id": "m"}}))
    respx.post(f"{BASE_URL}/v1/memory/recall").mock(
        return_value=httpx.Response(200, json={"memories": [{"memory": {"id": "m1", "content": "hi"}, "score": 0.9}]})
    )
    out = restored.run({"writer": {"messages": [ChatMessage.from_user("hello")]}, "retriever": {"query": "q"}})
    assert out["writer"]["memories_written"] == 1
    assert isinstance(out["retriever"]["memories"][0], ChatMessage)


def test_document_pipeline_dumps_loads_and_runs(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "dk-test")
    with patch("haystack_integrations.document_stores.dakera.document_store.DakeraClient") as mock_client:
        mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=8, vector_count=0)
        mock_client.return_value.query.return_value = SimpleNamespace(
            results=[SimpleNamespace(id="a", score=0.9, values=[0.1] * 8, metadata={"_dakera_content": "doc a"})]
        )
        store = DakeraDocumentStore(api_key=Secret.from_env_var("DAKERA_API_KEY"), namespace="ns", dimension=8)
        pipe = Pipeline()
        pipe.add_component("retriever", DakeraEmbeddingRetriever(document_store=store, top_k=1))

        restored = Pipeline.loads(pipe.dumps())
        out = restored.run({"retriever": {"query_embedding": [0.1] * 8}})
        assert out["retriever"]["documents"][0].content == "doc a"
