# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from haystack.dataclasses.document import Document

from haystack_integrations.components.retrievers.weaviate import (
    WeaviateBM25Retriever,
    WeaviateEmbeddingRetriever,
    WeaviateHybridRetriever,
)
from haystack_integrations.document_stores.weaviate.document_store import (
    DOCUMENT_COLLECTION_PROPERTIES,
    WeaviateDocumentStore,
)

# The retriever components were previously only exercised against mocks, so nothing checked that the
# calls they make actually work against Weaviate. These tests run the components end to end.

DOCUMENTS = [
    Document(content="The quick brown fox", meta={"category": "animals"}, embedding=[1.0, 0.0, 0.0, 0.0]),
    Document(content="A lazy dog sleeps", meta={"category": "animals"}, embedding=[0.0, 1.0, 0.0, 0.0]),
    Document(content="Functional programming in Haskell", meta={"category": "code"}, embedding=[0.0, 0.0, 1.0, 0.0]),
]


def _collection_settings(name: str) -> dict:
    return {
        "class": name,
        "invertedIndexConfig": {"indexNullState": True},
        "properties": [*DOCUMENT_COLLECTION_PROPERTIES, {"name": "category", "dataType": ["text"]}],
    }


@pytest.fixture
def document_store(request) -> Generator[WeaviateDocumentStore, None, None]:
    settings = _collection_settings(request.node.name.replace("[", "_").replace("]", ""))
    store = WeaviateDocumentStore(url="http://localhost:8080", collection_settings=settings)
    store.write_documents(DOCUMENTS)
    yield store
    store.client.collections.delete(settings["class"])
    store.close()


@pytest_asyncio.fixture
async def document_store_async(request) -> AsyncGenerator[WeaviateDocumentStore, None]:
    settings = _collection_settings(request.node.name.replace("[", "_").replace("]", "") + "Async")
    store = WeaviateDocumentStore(url="http://localhost:8080", collection_settings=settings)
    await store.write_documents_async(DOCUMENTS)
    yield store
    async_client = await store.async_client
    await async_client.collections.delete(settings["class"])
    await store.close_async()


@pytest.mark.integration
class TestWeaviateRetrieversIntegration:
    def test_bm25_retriever(self, document_store):
        result = WeaviateBM25Retriever(document_store=document_store).run(query="Haskell")

        assert [doc.content for doc in result["documents"]] == ["Functional programming in Haskell"]
        assert result["documents"][0].score is not None

    def test_bm25_retriever_honors_top_k_and_filters(self, document_store):
        retriever = WeaviateBM25Retriever(document_store=document_store, top_k=1)
        result = retriever.run(
            query="fox dog", filters={"field": "meta.category", "operator": "==", "value": "animals"}
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["category"] == "animals"

    def test_embedding_retriever(self, document_store):
        retriever = WeaviateEmbeddingRetriever(document_store=document_store, top_k=1)
        result = retriever.run(query_embedding=[0.0, 0.0, 1.0, 0.0])

        assert [doc.content for doc in result["documents"]] == ["Functional programming in Haskell"]

    def test_embedding_retriever_honors_filters(self, document_store):
        retriever = WeaviateEmbeddingRetriever(document_store=document_store, top_k=3)
        result = retriever.run(
            query_embedding=[0.0, 0.0, 1.0, 0.0],
            filters={"field": "meta.category", "operator": "==", "value": "animals"},
        )

        assert {doc.meta["category"] for doc in result["documents"]} == {"animals"}

    def test_hybrid_retriever(self, document_store):
        retriever = WeaviateHybridRetriever(document_store=document_store, top_k=2)
        result = retriever.run(query="Haskell", query_embedding=[0.0, 0.0, 1.0, 0.0])

        assert result["documents"][0].content == "Functional programming in Haskell"
        assert len(result["documents"]) == 2

    def test_hybrid_retriever_honors_filters(self, document_store):
        retriever = WeaviateHybridRetriever(document_store=document_store)
        result = retriever.run(
            query="fox",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.category", "operator": "==", "value": "code"},
        )

        assert {doc.meta["category"] for doc in result["documents"]} == {"code"}


@pytest.mark.integration
class TestWeaviateRetrieversAsyncIntegration:
    async def test_bm25_retriever_async(self, document_store_async):
        retriever = WeaviateBM25Retriever(document_store=document_store_async)
        result = await retriever.run_async(query="Haskell")

        assert [doc.content for doc in result["documents"]] == ["Functional programming in Haskell"]

    async def test_embedding_retriever_async(self, document_store_async):
        retriever = WeaviateEmbeddingRetriever(document_store=document_store_async, top_k=1)
        result = await retriever.run_async(query_embedding=[0.0, 0.0, 1.0, 0.0])

        assert [doc.content for doc in result["documents"]] == ["Functional programming in Haskell"]

    async def test_hybrid_retriever_async(self, document_store_async):
        retriever = WeaviateHybridRetriever(document_store=document_store_async, top_k=2)
        result = await retriever.run_async(query="Haskell", query_embedding=[0.0, 0.0, 1.0, 0.0])

        assert result["documents"][0].content == "Functional programming in Haskell"
