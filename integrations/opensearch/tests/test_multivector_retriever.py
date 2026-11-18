# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.opensearch import OpenSearchMultivectorEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def test_init_invalid_document_store():
    with pytest.raises(TypeError, match="OpenSearchDocumentStore"):
        OpenSearchMultivectorEmbeddingRetriever(document_store=None)


def test_init_requires_multivector_support():
    with pytest.raises(ValueError, match="multivector_embedding_dim"):
        OpenSearchMultivectorEmbeddingRetriever(document_store=OpenSearchDocumentStore())


def test_init_invalid_top_k():
    store = OpenSearchDocumentStore(multivector_embedding_dim=2)
    with pytest.raises(ValueError, match="top_k"):
        OpenSearchMultivectorEmbeddingRetriever(document_store=store, top_k=0)


def test_to_dict():
    store = OpenSearchDocumentStore(multivector_embedding_dim=2, return_embedding=True)
    retriever = OpenSearchMultivectorEmbeddingRetriever(document_store=store, top_k=3, filter_policy="merge")
    assert retriever.to_dict() == {
        "type": (
            "haystack_integrations.components.retrievers.opensearch.multivector_retriever."
            "OpenSearchMultivectorEmbeddingRetriever"
        ),
        "init_parameters": {"document_store": store.to_dict(), "filters": {}, "top_k": 3, "filter_policy": "merge"},
    }


def test_from_dict():
    data = {
        "type": (
            "haystack_integrations.components.retrievers.opensearch.multivector_retriever."
            "OpenSearchMultivectorEmbeddingRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                "init_parameters": {"multivector_embedding_dim": 2, "return_embedding": True},
            },
            "filters": {},
            "top_k": 3,
            "filter_policy": "merge",
        },
    }
    restored = OpenSearchMultivectorEmbeddingRetriever.from_dict(data)
    assert restored._top_k == 3
    assert restored._filter_policy.value == "merge"
    assert restored._document_store._multivector_embedding_dim == 2
    assert restored._document_store._return_embedding is True


def test_close():
    store = Mock(spec=OpenSearchDocumentStore, _multivector_embedding_dim=2)
    OpenSearchMultivectorEmbeddingRetriever(document_store=store).close()
    store.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_async():
    store = Mock(spec=OpenSearchDocumentStore, _multivector_embedding_dim=2)
    await OpenSearchMultivectorEmbeddingRetriever(document_store=store).close_async()
    store.close_async.assert_awaited_once_with()


def test_run():
    store = OpenSearchDocumentStore(multivector_embedding_dim=2)
    store._client = Mock()
    store._client.search.return_value = {
        "hits": {
            "hits": [
                {"_source": {"id": "match"}, "_score": 2.0},
                {"_source": {"id": "missing"}, "_score": 0.0},
            ]
        }
    }
    retriever = OpenSearchMultivectorEmbeddingRetriever(
        document_store=store,
        filters={"field": "meta.kind", "operator": "==", "value": "article"},
        filter_policy=FilterPolicy.MERGE,
    )
    result = retriever.run(
        query_multivector_embedding=[[1.0, 0.0]], top_k=2, filters={"field": "meta.page", "operator": "==", "value": 2}
    )
    assert result["documents"] == [Document(id="match", score=2.0)]
    body = store._client.search.call_args.kwargs["body"]
    assert body["size"] == 2
    script_score = body["query"]["script_score"]
    assert script_score["script"]["params"]["query_vectors"] == [[1.0, 0.0]]
    conditions = script_score["query"]["bool"]["filter"]["bool"]["must"]
    assert {"term": {"kind": "article"}} in conditions
    assert {"term": {"page": 2}} in conditions


def test_run_with_runtime_document_store():
    initial_store = Mock(spec=OpenSearchDocumentStore, _multivector_embedding_dim=2)
    runtime_store = Mock(spec=OpenSearchDocumentStore, _multivector_embedding_dim=2)
    runtime_store._multivector_retrieval.return_value = [Document(id="runtime")]
    retriever = OpenSearchMultivectorEmbeddingRetriever(document_store=initial_store)

    result = retriever.run(query_multivector_embedding=[[1.0, 0.0]], document_store=runtime_store)

    assert result["documents"] == [Document(id="runtime")]
    initial_store._multivector_retrieval.assert_not_called()


@pytest.mark.asyncio
async def test_run_async():
    store = OpenSearchDocumentStore(multivector_embedding_dim=2)
    store._async_client = AsyncMock()
    store._async_client.search.return_value = {"hits": {"hits": [{"_source": {"id": "match"}, "_score": 2.0}]}}
    retriever = OpenSearchMultivectorEmbeddingRetriever(document_store=store)
    result = await retriever.run_async(query_multivector_embedding=[[1.0, 0.0]])
    assert result["documents"] == [Document(id="match", score=2.0)]
