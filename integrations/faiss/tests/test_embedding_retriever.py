# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

EMBEDDING_DIM = 3


@pytest.fixture
def document_store():
    """In-memory FAISSDocumentStore with dim=3 for fast unit tests."""
    return FAISSDocumentStore(embedding_dim=EMBEDDING_DIM)


@pytest.fixture
def populated_store(document_store):
    """Store pre-loaded with 3 documents that have embeddings and metadata."""
    docs = [
        Document(content="alpha", embedding=[1.0, 0.0, 0.0], meta={"category": "A"}),
        Document(content="beta", embedding=[0.0, 1.0, 0.0], meta={"category": "B"}),
        Document(content="gamma", embedding=[0.0, 0.0, 1.0], meta={"category": "A"}),
    ]
    document_store.write_documents(docs)
    return document_store


class TestFAISSEmbeddingRetriever:
    def test_run_with_query_embedding_only(self, populated_store):
        retriever = FAISSEmbeddingRetriever(document_store=populated_store, top_k=2)
        result = retriever.run(query_embedding=[1.0, 0.0, 0.0])

        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 2
        # All returned items must be Document instances
        assert all(isinstance(d, Document) for d in result["documents"])

    def test_run_with_filters(self, populated_store):
        retriever = FAISSEmbeddingRetriever(document_store=populated_store, top_k=3)
        filters = {"field": "meta.category", "operator": "==", "value": "A"}
        result = retriever.run(query_embedding=[1.0, 0.0, 0.0], filters=filters)

        assert "documents" in result
        contents = [d.content for d in result["documents"]]
        # Only category-A docs should be returned
        assert all(d.meta["category"] == "A" for d in result["documents"])
        assert "beta" not in contents

    def test_run_with_top_k_override(self, populated_store):
        retriever = FAISSEmbeddingRetriever(document_store=populated_store, top_k=3)
        result = retriever.run(query_embedding=[1.0, 0.0, 0.0], top_k=1)

        assert len(result["documents"]) == 1

    def test_to_dict_from_dict_roundtrip(self, document_store):
        retriever = FAISSEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
        )

        serialized = retriever.to_dict()
        assert serialized["type"] == (
            "haystack_integrations.components.retrievers.faiss.embedding_retriever.FAISSEmbeddingRetriever"
        )
        assert serialized["init_parameters"]["top_k"] == 5
        assert serialized["init_parameters"]["filter_policy"] == FilterPolicy.MERGE.value
        assert "document_store" in serialized["init_parameters"]

        restored = FAISSEmbeddingRetriever.from_dict(serialized)
        assert restored.top_k == 5
        assert restored.filter_policy == FilterPolicy.MERGE
        assert isinstance(restored.document_store, FAISSDocumentStore)

    def test_filter_policy_replace(self, populated_store):
        """REPLACE: runtime filters fully replace init-time filters."""
        init_filters = {"field": "meta.category", "operator": "==", "value": "A"}
        runtime_filters = {"field": "meta.category", "operator": "==", "value": "B"}

        retriever = FAISSEmbeddingRetriever(
            document_store=populated_store,
            filters=init_filters,
            top_k=3,
            filter_policy=FilterPolicy.REPLACE,
        )
        result = retriever.run(query_embedding=[0.0, 1.0, 0.0], filters=runtime_filters)

        # Only category B docs should appear — the init filter was replaced
        assert all(d.meta["category"] == "B" for d in result["documents"])

    def test_filter_policy_merge(self, populated_store):
        """MERGE: runtime filters are merged with init-time filters."""
        init_filters = {"field": "meta.category", "operator": "==", "value": "A"}

        retriever = FAISSEmbeddingRetriever(
            document_store=populated_store,
            filters=init_filters,
            top_k=3,
            filter_policy=FilterPolicy.MERGE,
        )
        # Run without any runtime filter — init filter alone should apply
        result = retriever.run(query_embedding=[1.0, 0.0, 0.0])

        assert len(result["documents"]) >= 1
        assert all(d.meta["category"] == "A" for d in result["documents"])

    def test_invalid_document_store_type(self):
        with pytest.raises(ValueError, match="document_store must be an instance of FAISSDocumentStore"):
            FAISSEmbeddingRetriever(document_store="not_a_store")  # type: ignore[arg-type]

    def test_run_in_pipeline(self, populated_store):
        """End-to-end: FAISSEmbeddingRetriever wired into a Haystack Pipeline."""
        retriever = FAISSEmbeddingRetriever(document_store=populated_store, top_k=2)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)

        result = pipeline.run({"retriever": {"query_embedding": [1.0, 0.0, 0.0]}})

        assert "retriever" in result
        assert "documents" in result["retriever"]
        assert isinstance(result["retriever"]["documents"], list)
        assert len(result["retriever"]["documents"]) == 2
        assert all(isinstance(d, Document) for d in result["retriever"]["documents"])
