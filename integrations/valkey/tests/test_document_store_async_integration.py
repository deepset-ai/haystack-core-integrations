import pytest
import pytest_asyncio
import uuid
from haystack.dataclasses import Document, ByteStream
from haystack_integrations.document_stores.valkey import ValkeyDocumentStore, ValkeyDocumentStoreError


@pytest.mark.integration
@pytest.mark.asyncio
class TestValkeyDocumentStoreAsyncIntegration:
    @pytest_asyncio.fixture
    async def document_store(self):
        store = ValkeyDocumentStore(index_name="test_async_haystack_document", embedding_dim=3, batch_size=5)
        yield store
        await store.async_close()

    @pytest_asyncio.fixture(autouse=True)
    async def cleanup_after_test(self, document_store):
        yield
        # Clean up all data after each test
        try:
            await document_store._async_client.flushdb()
        except Exception:
            pass

    async def test_async_write_and_count_documents(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(id=f"async1_{test_id}", content="async test doc 1", embedding=[0.1, 0.2, 0.3]),
            Document(id=f"async2_{test_id}", content="async test doc 2", embedding=[0.4, 0.5, 0.6]),
            Document(id=f"async3_{test_id}", content="async test doc 3"),  # No embedding
        ]

        result = await document_store.async_write_documents(docs)
        assert result == 3

        count = await document_store.async_count_documents()
        assert count == 3

    async def test_async_write_exceed_batch_size(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(id=f"async1_{test_id}", content="async test doc 1", embedding=[0.1, 0.2, 0.3]),
            Document(id=f"async2_{test_id}", content="async test doc 2", embedding=[0.4, 0.5, 0.6]),
            Document(id=f"async3_{test_id}", content="async test doc 3"),  # No embedding
            Document(id=f"async4_{test_id}", content="async test doc 4", embedding=[0.1, 0.2, 0.3]),
            Document(id=f"async5_{test_id}", content="async test doc 5", embedding=[0.4, 0.5, 0.6]),
            Document(id=f"async6_{test_id}", content="async test doc 6"),  # No embedding
        ]

        result = await document_store.async_write_documents(docs)
        assert result == 6

        count = await document_store.async_count_documents()
        assert count == 6

    async def test_async_write_exact_batch_size(self, document_store):
        """Test writing exactly batch_size documents (5)"""
        test_id = str(uuid.uuid4())[:8]
        docs = [Document(id=f"exact_{i}_{test_id}", content=f"doc {i}") for i in range(5)]

        result = await document_store.async_write_documents(docs)
        assert result == 5

        count = await document_store.async_count_documents()
        assert count == 5

    async def test_async_write_multiple_full_batches(self, document_store):
        """Test writing multiple full batches (11 docs = 2 full + 1 partial)"""
        test_id = str(uuid.uuid4())[:8]
        docs = [Document(id=f"multi_{i}_{test_id}", content=f"doc {i}") for i in range(11)]

        result = await document_store.async_write_documents(docs)
        assert result == 11

        count = await document_store.async_count_documents()
        assert count == 11

    async def test_async_write_and_delete_documents(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(id=f"async_del1_{test_id}", content="async delete me", embedding=[0.1, 0.2, 0.3]),
            Document(id=f"async_del2_{test_id}", content="async delete me too", embedding=[0.4, 0.5, 0.6]),
        ]

        await document_store.async_write_documents(docs)
        count = await document_store.async_count_documents()
        assert count == 2

        await document_store.async_delete_documents([f"async_del1_{test_id}", f"async_del2_{test_id}"])
        count = await document_store.async_count_documents()
        assert count == 0

    async def test_async_overwrite_documents(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        doc1 = Document(id=f"async_overwrite_{test_id}", content="async original", embedding=[0.1, 0.2, 0.3])

        await document_store.async_write_documents([doc1])

        doc2 = Document(id=f"async_overwrite_{test_id}", content="async updated", embedding=[0.4, 0.5, 0.6])
        result = await document_store.async_write_documents([doc2])

        assert result == 1
        count = await document_store.async_count_documents()
        assert count == 1

    async def test_async_search_by_embedding_no_limit(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"search1_{test_id}",
                content="similar content",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "test", "priority": 1},
                blob=ByteStream(data=b"binary_data", mime_type="application/octet-stream"),
                score=0.95,
            ),
            Document(
                id=f"search2_{test_id}",
                content="different content",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "other", "priority": 2},
                score=0.85,
            ),
            Document(
                id=f"search3_{test_id}",
                content="another content",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "test", "priority": 3},
            ),
            Document(id=f"search4_{test_id}", content="another content", meta={"category": "test", "priority": 3}),
        ]

        await document_store.async_write_documents(docs)

        # Verify documents are written
        count = await document_store.async_count_documents()
        assert count == 4

        # Search with embedding similar to first document
        query_embedding = [0.1, 0.2, 0.3]
        results = await document_store.async_search(query_embedding, limit=100)

        assert len(results) == 4
        assert results[0].id == f"search1_{test_id}"  # Most similar should be first
        assert results[3].id == f"search4_{test_id}"  # Document without embedding should be last

    async def test_async_search_by_embedding_with_limit(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"search1_{test_id}",
                content="similar content",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "test", "priority": 1},
                blob=ByteStream(data=b"binary_data", mime_type="application/octet-stream"),
                score=0.95,
            ),
            Document(
                id=f"search2_{test_id}",
                content="different content",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "other", "priority": 2},
                score=0.85,
            ),
            Document(
                id=f"search3_{test_id}",
                content="another content",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "test", "priority": 3},
            ),
            Document(id=f"search4_{test_id}", content="another content", meta={"category": "test", "priority": 3}),
        ]

        await document_store.async_write_documents(docs)

        # Verify documents are written
        count = await document_store.async_count_documents()
        assert count == 4

        # Search with embedding similar to first document
        query_embedding = [0.1, 0.2, 0.3]
        results = await document_store.async_search(query_embedding, limit=2)

        assert len(results) == 2
        assert results[0].id == f"search1_{test_id}"  # Most similar should be first
        assert results[1].id == f"search3_{test_id}"

    async def test_async_search_by_embedding_with_category_filter(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"search1_{test_id}",
                content="similar content",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "test", "priority": 1},
                blob=ByteStream(data=b"binary_data", mime_type="application/octet-stream"),
                score=0.95,
            ),
            Document(
                id=f"search2_{test_id}",
                content="different content",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "other", "priority": 2},
                score=0.85,
            ),
            Document(
                id=f"search3_{test_id}",
                content="another content",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "test2", "priority": 3},
            ),
            Document(id=f"search4_{test_id}", content="another content", meta={"category": "test3", "priority": 3}),
        ]

        await document_store.async_write_documents(docs)

        # Verify documents are written
        count = await document_store.async_count_documents()
        assert count == 4

        # Search with embedding similar to first document
        query_embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "test"}]}
        results = await document_store.async_search(query_embedding, filters, limit=2)

        assert len(results) == 1
        assert results[0].id == f"search1_{test_id}"  # Most similar should be first

    async def test_async_search_by_embedding_with_numeric_filter(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"n1_{test_id}", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"priority": 1, "score": 0.8}
            ),
            Document(
                id=f"n2_{test_id}", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"priority": 5, "score": 0.9}
            ),
            Document(
                id=f"n3_{test_id}", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"priority": 10, "score": 0.7}
            ),
        ]
        await document_store.async_write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": ">=", "value": 5}]}
        results = await document_store.async_search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {f"n2_{test_id}", f"n3_{test_id}"}
        # Results should be ordered by similarity to query
        assert results[0].id == f"n2_{test_id}"  # Closer to query embedding

    async def test_async_search_by_embedding_with_or_filter(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"o1_{test_id}",
                content="doc 1",
                embedding=[0.1, 0.2, 0.3],
                meta={"status": "active", "category": "news"},
            ),
            Document(
                id=f"o2_{test_id}",
                content="doc 2",
                embedding=[0.2, 0.3, 0.4],
                meta={"status": "pending", "category": "sports"},
            ),
            Document(
                id=f"o3_{test_id}",
                content="doc 3",
                embedding=[0.9, 0.8, 0.7],
                meta={"status": "inactive", "category": "news"},
            ),
        ]
        await document_store.async_write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.status", "operator": "==", "value": "active"},
                {"field": "meta.category", "operator": "==", "value": "sports"},
            ],
        }
        results = await document_store.async_search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {f"o1_{test_id}", f"o2_{test_id}"}
        # Results should be ordered by similarity
        assert results[0].id == f"o1_{test_id}"  # Closer to query embedding

    async def test_async_search_by_embedding_with_in_filter(self, document_store):
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(id=f"i1_{test_id}", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"status": "active"}),
            Document(id=f"i2_{test_id}", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"status": "pending"}),
            Document(id=f"i3_{test_id}", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"status": "inactive"}),
            Document(id=f"i4_{test_id}", content="doc 4", embedding=[0.9, 0.8, 0.7], meta={"status": "archived"}),
        ]
        await document_store.async_write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "AND",
            "conditions": [{"field": "meta.status", "operator": "in", "value": ["active", "pending"]}],
        }
        results = await document_store.async_search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {f"i1_{test_id}", f"i2_{test_id}"}
        # Results should be ordered by similarity
        assert results[0].id == f"i1_{test_id}"  # Closest to query

    async def test_async_delete_all_documents(self, document_store):
        docs = [
            Document(id="del1", content="doc 1", embedding=[0.1, 0.2, 0.3]),
            Document(id="del2", content="doc 2", embedding=[0.4, 0.5, 0.6]),
            Document(id="del3", content="doc 3", embedding=[0.7, 0.8, 0.9]),
        ]

        await document_store.async_write_documents(docs)
        assert await document_store.async_count_documents() == 3

        await document_store.async_delete_all_documents()
        assert await document_store.async_count_documents() == 0

    async def test_async_delete_all_documents_empty_store(self, document_store):
        assert await document_store.async_count_documents() == 0

        await document_store.async_delete_all_documents()
        assert await document_store.async_count_documents() == 0

    async def test_async_write_large_batch_performance(self, document_store):
        """Test writing large number of documents to verify batching performance"""
        test_id = str(uuid.uuid4())[:8]
        # 23 documents = 4 full batches (5 each) + 1 partial batch (3)
        docs = [Document(id=f"perf_{i}_{test_id}", content=f"performance doc {i}") for i in range(23)]

        result = await document_store.async_write_documents(docs)
        assert result == 23

        count = await document_store.async_count_documents()
        assert count == 23

    async def test_async_similarity_scores_are_set_correctly(self, document_store):
        """Test that similarity scores are properly computed and set for all returned documents in async mode."""
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(id=f"sim1_{test_id}", content="identical vector", embedding=[1.0, 0.0, 0.0]),
            Document(id=f"sim2_{test_id}", content="similar vector", embedding=[0.9, 0.1, 0.0]),
            Document(id=f"sim3_{test_id}", content="different vector", embedding=[0.0, 1.0, 0.0]),
            Document(id=f"sim4_{test_id}", content="opposite vector", embedding=[-1.0, 0.0, 0.0]),
        ]

        await document_store.async_write_documents(docs)

        # Search with query identical to first document
        query_embedding = [1.0, 0.0, 0.0]
        results = await document_store.async_search(query_embedding, limit=10)

        # All documents should have similarity scores set
        assert len(results) == 4
        for doc in results:
            assert doc.score is not None, f"Document {doc.id} has no similarity score"
            assert isinstance(doc.score, float), f"Document {doc.id} score is not a float: {type(doc.score)}"

        # Results should be ordered by similarity (highest first for cosine similarity)
        assert results[0].id == f"sim1_{test_id}"  # Identical vector should have highest score
        assert results[1].id == f"sim2_{test_id}"  # Similar vector should be second

        # Verify scores are properly computed (not just dummy values)
        scores = [doc.score for doc in results]
        assert len(set(scores)) > 1, "All similarity scores are identical, suggesting they're not properly computed"

        # Verify all results are sorted by similarity score (lower is better for distance metrics)
        for i in range(len(results) - 1):
            assert results[i].score <= results[i + 1].score, (
                f"Results not sorted by score: {results[i].score} > {results[i + 1].score} at positions {i} and {i + 1}"
            )

        # The identical vector should have the lowest (best) similarity score
        assert results[0].score <= results[1].score, (
            f"Expected identical vector to have best score, got {results[0].score} vs {results[1].score}"
        )

    async def test_async_filter_by_meta_score(self, document_store):
        """Test filtering by user-provided meta.score values in async mode."""
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"ms1_{test_id}", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"score": 0.9, "category": "high"}
            ),
            Document(
                id=f"ms2_{test_id}",
                content="doc 2",
                embedding=[0.2, 0.3, 0.4],
                meta={"score": 0.5, "category": "medium"},
            ),
            Document(
                id=f"ms3_{test_id}", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"score": 0.2, "category": "low"}
            ),
            Document(
                id=f"ms4_{test_id}", content="doc 4", embedding=[0.4, 0.5, 0.6], meta={"score": 0.8, "category": "high"}
            ),
        ]
        await document_store.async_write_documents(docs)

        # Filter by meta.score >= 0.7
        filters = {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": 0.7}]}
        results = await document_store.async_filter_documents(filters)

        assert len(results) == 2
        result_ids = {doc.id for doc in results}
        assert result_ids == {f"ms1_{test_id}", f"ms4_{test_id}"}

        # Verify meta.score values are preserved
        for doc in results:
            assert "score" in doc.meta
            assert doc.meta["score"] >= 0.7

    async def test_async_search_with_meta_score_filter(self, document_store):
        """Test vector search combined with meta.score filtering in async mode."""
        test_id = str(uuid.uuid4())[:8]
        docs = [
            Document(
                id=f"sms1_{test_id}",
                content="doc 1",
                embedding=[1.0, 0.0, 0.0],
                meta={"score": 0.9, "quality": "excellent"},
            ),
            Document(
                id=f"sms2_{test_id}", content="doc 2", embedding=[0.9, 0.1, 0.0], meta={"score": 0.3, "quality": "poor"}
            ),
            Document(
                id=f"sms3_{test_id}", content="doc 3", embedding=[0.8, 0.2, 0.0], meta={"score": 0.8, "quality": "good"}
            ),
            Document(
                id=f"sms4_{test_id}",
                content="doc 4",
                embedding=[0.0, 1.0, 0.0],
                meta={"score": 0.95, "quality": "excellent"},
            ),
        ]
        await document_store.async_write_documents(docs)

        # Search with query similar to first document, but filter by meta.score
        query_embedding = [1.0, 0.0, 0.0]
        filters = {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": 0.7}]}
        results = await document_store.async_search(query_embedding, filters, limit=10)

        # Should return documents with meta.score >= 0.7, ordered by vector similarity
        assert len(results) == 3
        result_ids = [doc.id for doc in results]
        assert set(result_ids) == {f"sms1_{test_id}", f"sms3_{test_id}", f"sms4_{test_id}"}

        # Verify ordering by vector similarity (sms1 should be first as it's most similar)
        assert results[0].id == f"sms1_{test_id}"

        # Verify both similarity scores and meta scores are preserved
        for doc in results:
            assert doc.score is not None  # Vector similarity score
            assert "score" in doc.meta  # User metadata score
            assert doc.meta["score"] >= 0.7
