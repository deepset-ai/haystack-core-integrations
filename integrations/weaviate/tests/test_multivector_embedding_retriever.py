# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import create_autospec

import pytest
from haystack import Document, Pipeline
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.weaviate import WeaviateMultivectorEmbeddingRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.document_stores.weaviate.auth import AuthApiKey


def test_serialization_roundtrip(monkeypatch):
    monkeypatch.setenv("WEAVIATE_API_KEY", "test-key")
    retriever = WeaviateMultivectorEmbeddingRetriever(
        document_store=WeaviateDocumentStore(
            url="http://localhost:8080",
            auth_client_secret=AuthApiKey(),
        ),
        filters={"field": "meta.category", "operator": "==", "value": "test"},
        top_k=3,
        distance=0.5,
        filter_policy=FilterPolicy.MERGE,
    )
    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    restored = Pipeline.loads(pipeline.dumps()).get_component("retriever")
    assert restored.to_dict() == retriever.to_dict()
    assert restored._document_store._auth_client_secret.resolve_value().api_key == "test-key"


@pytest.mark.parametrize(
    "filter_policy,override", [(FilterPolicy.REPLACE, False), (FilterPolicy.REPLACE, True), (FilterPolicy.MERGE, True)]
)
def test_run(filter_policy, override):
    store = create_autospec(WeaviateDocumentStore, instance=True)
    store._multivector_embedding_retrieval.return_value = [Document(content="result")]
    initial = {"field": "meta.category", "operator": "==", "value": "test"}
    runtime = {"field": "meta.page", "operator": "==", "value": 1}
    retriever = WeaviateMultivectorEmbeddingRetriever(
        document_store=store, filters=initial, top_k=5, distance=0.7, filter_policy=filter_policy
    )
    kwargs = {"filters": runtime, "top_k": 2, "distance": 0.0} if override else {}
    result = retriever.run(query_multivector_embedding=[[1.0, 0.0]], **kwargs)
    expected_filters = initial
    if override:
        expected_filters = (
            runtime if filter_policy == FilterPolicy.REPLACE else {"operator": "AND", "conditions": [initial, runtime]}
        )
    store._multivector_embedding_retrieval.assert_called_once_with(
        query_multivector_embedding=[[1.0, 0.0]],
        filters=expected_filters,
        top_k=2 if override else 5,
        distance=0.0 if override else 0.7,
    )
    assert result == {"documents": store._multivector_embedding_retrieval.return_value}


def test_close():
    store = create_autospec(WeaviateDocumentStore, instance=True)
    WeaviateMultivectorEmbeddingRetriever(document_store=store).close()
    store.close.assert_called_once_with()


@pytest.mark.integration
def test_retrieval(multivector_document_store):
    query = [[1.0, 0.0], [1.0, 0.0]]
    store = multivector_document_store
    store.write_documents(
        [
            Document(id="match", content="matching", multivector_embedding=[[1.0, 0.0]], meta={"category": "keep"}),
            Document(id="other", content="other", multivector_embedding=[[0.0, 1.0]], meta={"category": "other"}),
        ]
    )
    retriever = WeaviateMultivectorEmbeddingRetriever(document_store=store, top_k=1)
    docs = retriever.run(query_multivector_embedding=query)["documents"]
    assert [doc.id for doc in docs] == ["match"]
    assert docs[0].multivector_embedding == [[1.0, 0.0]]
    assert docs[0].score == pytest.approx(len(query))
    docs = retriever.run(
        query_multivector_embedding=query,
        filters={"field": "meta.category", "operator": "==", "value": "other"},
    )["documents"]
    assert [doc.id for doc in docs] == ["other"]
    docs = retriever.run(query_multivector_embedding=query, top_k=2, distance=-0.5)["documents"]
    assert [doc.id for doc in docs] == ["match"]
