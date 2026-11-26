# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import create_autospec

import pytest
from haystack import Document

from haystack_integrations.components.retrievers.weaviate import WeaviateHybridRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "multivector", "combined"])
async def test_named_vector_hybrid_retrieval_async(multivector_document_store, mode):
    store = multivector_document_store
    try:
        await store.write_documents_async(
            [
                Document(
                    id="match",
                    content="matching document",
                    embedding=[1.0, 0.0],
                    multivector_embedding=[[1.0, 0.0]],
                    meta={"category": "keep"},
                ),
                Document(
                    id="other",
                    content="other document",
                    embedding=[0.0, 1.0],
                    multivector_embedding=[[0.0, 1.0]],
                    meta={"category": "other"},
                ),
            ]
        )
        inputs = {}
        if mode in ("dense", "combined"):
            inputs["query_embedding"] = [1.0, 0.0]
        if mode in ("multivector", "combined"):
            inputs["query_multivector_embedding"] = [[1.0, 0.0]]
        retriever = WeaviateHybridRetriever(
            document_store=store, top_k=1, vector_weights={"default": 0.2, "multi_vector": 0.8}
        )
        docs = (await retriever.run_async(query="matching", **inputs))["documents"]
        assert [doc.id for doc in docs] == ["match"]
        assert docs[0].embedding == [1.0, 0.0]
        assert docs[0].multivector_embedding == [[1.0, 0.0]]
        assert docs[0].score is not None
        docs = (
            await retriever.run_async(
                query="document",
                **inputs,
                filters={"field": "meta.category", "operator": "==", "value": "other"},
            )
        )["documents"]
        assert [doc.id for doc in docs] == ["other"]
    finally:
        await store.close_async()


@pytest.mark.parametrize("override", [False, True])
async def test_combined_inputs_async(override):
    store = create_autospec(WeaviateDocumentStore, instance=True)
    store._hybrid_retrieval_async.return_value = [Document(content="result")]
    initial = {"default": 0.2, "multi_vector": 0.8}
    runtime = {"default": 0.8, "multi_vector": 0.2}
    retriever = WeaviateHybridRetriever(document_store=store, vector_weights=initial)
    result = await retriever.run_async(
        query="test",
        query_embedding=[1.0, 0.0],
        query_multivector_embedding=[[0.0, 1.0]],
        vector_weights=runtime if override else None,
    )
    store._hybrid_retrieval_async.assert_awaited_once_with(
        query="test",
        query_embedding=[1.0, 0.0],
        query_multivector_embedding=[[0.0, 1.0]],
        vector_weights=runtime if override else initial,
        filters={},
        top_k=10,
        alpha=0.7,
        max_vector_distance=None,
    )
    assert result == {"documents": store._hybrid_retrieval_async.return_value}


async def test_requires_embedding_async():
    store = create_autospec(WeaviateDocumentStore, instance=True)
    with pytest.raises(ValueError, match="Provide query_embedding"):
        await WeaviateHybridRetriever(document_store=store).run_async(query="test")
    store._hybrid_retrieval_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_async():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)

    await retriever.close_async()

    mock_document_store.close_async.assert_awaited_once_with()
    assert retriever._document_store is mock_document_store


@pytest.mark.asyncio
async def test_run_async_calls_async_retrieval():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    mock_document_store._hybrid_retrieval_async.return_value = []

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)

    await retriever.run_async(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
        alpha=0.7,
        max_vector_distance=0.5,
    )

    mock_document_store._hybrid_retrieval_async.assert_called_once_with(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
        alpha=0.7,
        max_vector_distance=0.5,
        query_multivector_embedding=None,
        vector_weights=None,
    )


@pytest.mark.asyncio
async def test_run_async_with_init_parameters():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    mock_document_store._hybrid_retrieval_async.return_value = []

    retriever = WeaviateHybridRetriever(
        document_store=mock_document_store,
        top_k=20,
        alpha=0.3,
        max_vector_distance=0.9,
    )

    await retriever.run_async(query="test", query_embedding=[0.1, 0.2, 0.3])

    mock_document_store._hybrid_retrieval_async.assert_called_once_with(
        query="test",
        query_embedding=[0.1, 0.2, 0.3],
        filters={},
        top_k=20,
        alpha=0.3,
        max_vector_distance=0.9,
        query_multivector_embedding=None,
        vector_weights=None,
    )


@pytest.mark.asyncio
async def test_run_async_with_invalid_alpha():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    with pytest.raises(ValueError, match=r"alpha \(-0.1\) must be in the range \[0.0, 1.0\]"):
        await retriever.run_async(query="q", query_embedding=[0.1], alpha=-0.1)
    with pytest.raises(ValueError, match=r"alpha \(1.5\) must be in the range \[0.0, 1.0\]"):
        await retriever.run_async(query="q", query_embedding=[0.1], alpha=1.5)
