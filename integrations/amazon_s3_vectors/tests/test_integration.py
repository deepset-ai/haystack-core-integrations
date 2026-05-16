# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Retriever-focused integration tests.

The Document Store contract is covered by `TestDocumentStore` / `TestFilters`
in `test_document_store.py` and `test_filters.py` (which inherit Haystack's
base test mixins). The tests here are the small set that don't fit those
mixins: the embedding-retrieval path and the `S3VectorsEmbeddingRetriever`
component wired up end-to-end.
"""

from __future__ import annotations

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


@pytest.mark.integration
def test_embedding_retrieval(document_store: S3VectorsDocumentStore) -> None:
    docs = [
        Document(id="r-1", content="First", embedding=[0.1] * 768),
        Document(id="r-2", content="Second", embedding=[0.5] * 768),
        Document(id="r-3", content="Third", embedding=[0.9] * 768),
    ]
    document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

    results = document_store._embedding_retrieval(query_embedding=[0.1] * 768, top_k=10)
    assert len(results) > 0
    for doc in results:
        assert doc.score is not None
        assert doc.content is not None
        # query_vectors does not return vector data
        assert doc.embedding is None


@pytest.mark.integration
def test_embedding_retrieval_with_metadata_filter(document_store: S3VectorsDocumentStore) -> None:
    docs = [
        Document(id="f-1", content="Sports article", embedding=[0.1] * 768, meta={"category": "sports"}),
        Document(id="f-2", content="News article", embedding=[0.1] * 768, meta={"category": "news"}),
    ]
    document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

    filters = {"field": "meta.category", "operator": "==", "value": "sports"}
    results = document_store._embedding_retrieval(query_embedding=[0.1] * 768, filters=filters, top_k=10)
    assert len(results) >= 1
    assert all(d.meta.get("category") == "sports" for d in results)


@pytest.mark.integration
def test_retriever_component(document_store: S3VectorsDocumentStore) -> None:
    docs = [Document(id="c-1", content="Hello", embedding=[0.1] * 768)]
    document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

    retriever = S3VectorsEmbeddingRetriever(document_store=document_store, top_k=5)
    result = retriever.run(query_embedding=[0.1] * 768)
    assert "documents" in result
    assert len(result["documents"]) > 0


@pytest.mark.integration
def test_to_from_dict_roundtrip(document_store: S3VectorsDocumentStore) -> None:
    """Serialization roundtrip against a real, fully-initialised store."""
    d = document_store.to_dict()
    restored = S3VectorsDocumentStore.from_dict(d)
    assert restored.vector_bucket_name == document_store.vector_bucket_name
    assert restored.index_name == document_store.index_name
    assert restored.dimension == document_store.dimension
    assert restored.distance_metric == document_store.distance_metric
