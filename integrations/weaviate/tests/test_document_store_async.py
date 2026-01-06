# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses.document import Document

from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.document_stores.weaviate.document_store import DOCUMENT_COLLECTION_PROPERTIES


@pytest.mark.integration
class TestWeaviateDocumentStoreAsync:
    @pytest.fixture
    def document_store(self, request) -> WeaviateDocumentStore:
        collection_settings = {
            "class": f"{request.node.name}",
            "invertedIndexConfig": {"indexNullState": True},
            "properties": DOCUMENT_COLLECTION_PROPERTIES,
        }
        store = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        yield store
        store.client.collections.delete(collection_settings["class"])

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Python is an object oriented programming language"),
                Document(content="Rust is a systems programming language"),
            ]
        )

        result = await document_store._bm25_retrieval_async("functional Haskell", top_k=2)

        assert len(result) <= 2
        assert any("Haskell" in doc.content for doc in result)
        assert all(doc.score is not None and doc.score > 0.0 for doc in result)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async_with_filters(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Python is an object oriented programming language"),
            ]
        )

        filters = {"field": "content", "operator": "==", "value": "Haskell"}
        result = await document_store._bm25_retrieval_async("functional", filters=filters)

        assert len(result) == 1
        assert "Haskell" in result[0].content

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async(self, document_store):
        document_store.write_documents(
            [
                Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
                Document(content="Yet another document", embedding=[0.00001, 0.00001, 0.00001, 0.00002]),
            ]
        )

        result = await document_store._embedding_retrieval_async(query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=2)

        assert len(result) == 2
        assert result[0].content == "The document"
        assert result[0].score is not None and result[0].score > 0.0

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_with_filters(self, document_store):
        document_store.write_documents(
            [
                Document(content="The document I want", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            ]
        )

        filters = {"field": "content", "operator": "==", "value": "The document I want"}
        result = await document_store._embedding_retrieval_async(query_embedding=[1.0, 1.0, 1.0, 1.0], filters=filters)

        assert len(result) == 1
        assert result[0].content == "The document I want"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_distance_and_certainty_error(self, document_store):
        with pytest.raises(ValueError, match="Can't use 'distance' and 'certainty' parameters together"):
            await document_store._embedding_retrieval_async(
                query_embedding=[1.0, 1.0, 1.0, 1.0], distance=0.5, certainty=0.8
            )

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_async(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="Python is an object oriented language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )

        result = await document_store._hybrid_retrieval_async(
            query="functional Haskell",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            top_k=2,
        )

        assert len(result) <= 2
        assert result[0].content == "Haskell is a functional programming language"
        assert result[0].score is not None and result[0].score > 0.0

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_async_with_filters(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
            ]
        )

        filters = {"field": "content", "operator": "==", "value": "Haskell is a functional programming language"}
        result = await document_store._hybrid_retrieval_async(
            query="functional",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            filters=filters,
        )

        assert len(result) == 1
        assert result[0].content == "Haskell is a functional programming language"

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_async_with_alpha(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Python is an object oriented language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )

        # Test with alpha=0.0 (pure BM25)
        result_bm25 = await document_store._hybrid_retrieval_async(
            query="functional",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            alpha=0.0,
        )
        assert len(result_bm25) > 0
        assert result_bm25[0].score > 0.0

        # Test with alpha=1.0 (pure vector search)
        result_vector = await document_store._hybrid_retrieval_async(
            query="functional",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            alpha=1.0,
        )
        assert len(result_vector) > 0
        assert result_vector[0].score > 0.0

    @pytest.mark.asyncio
    async def test_delete_by_filter_async(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Delete documents with category="A"
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert document_store.count_documents() == 1

        # Verify only category B remains
        remaining_docs = document_store.filter_documents()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    @pytest.mark.asyncio
    async def test_update_by_filter_async(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Update status for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 2

        # Verify the updates
        published_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

        # Verify category B still has draft status
        draft_docs = document_store.filter_documents(filters={"field": "meta.status", "operator": "==", "value": "draft"})
        assert len(draft_docs) == 1
        assert draft_docs[0].meta["category"] == "B"
