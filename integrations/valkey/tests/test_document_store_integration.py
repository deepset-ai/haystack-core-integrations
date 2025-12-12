# ruff: noqa: S110
import pytest
from haystack.dataclasses import Document
from haystack.dataclasses.byte_stream import ByteStream

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


@pytest.mark.integration
class TestValkeyDocumentStoreIntegration:
    @pytest.fixture
    def document_store(self):
        store = ValkeyDocumentStore(index_name="test_haystack_document", embedding_dim=3)
        yield store
        # Cleanup
        try:
            store.close()
        except Exception:
            pass

    @pytest.fixture(autouse=True)
    def cleanup_after_test(self, document_store):
        yield
        # Clean up all data after each test
        try:
            document_store._client.flushdb()
        except Exception:
            pass

    def test_write_and_count_documents(self, document_store):
        docs = [
            Document(id="1", content="test doc 1", embedding=[0.1, 0.2, 0.3]),
            Document(id="2", content="test doc 2", embedding=[0.4, 0.5, 0.6]),
            Document(id="3", content="test doc 3"),  # No embedding
        ]

        result = document_store.write_documents(docs)
        assert result == 3

        count = document_store.count_documents()
        assert count == 3

    def test_write_and_delete_documents(self, document_store):
        docs = [
            Document(id="del1", content="delete me", embedding=[0.1, 0.2, 0.3]),
            Document(id="del2", content="delete me too", embedding=[0.4, 0.5, 0.6]),
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_documents(["del1", "del2"])
        assert document_store.count_documents() == 0

    def test_overwrite_documents(self, document_store):
        doc1 = Document(id="overwrite", content="original", embedding=[0.1, 0.2, 0.3])
        document_store.write_documents([doc1])

        doc2 = Document(id="overwrite", content="updated", embedding=[0.4, 0.5, 0.6])
        result = document_store.write_documents([doc2])

        assert result == 1
        assert document_store.count_documents() == 1

    def test_search_by_embedding_no_limit(self, document_store):
        docs = [
            Document(
                id="search1",
                content="similar content",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "test", "priority": 1},
                blob=ByteStream(data=b"binary_data", mime_type="application/octet-stream"),
                score=0.95,
            ),
            Document(
                id="search2",
                content="different content",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "other", "priority": 2},
                score=0.85,
            ),
            Document(
                id="search3",
                content="another content",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "test", "priority": 3},
            ),
            Document(id="search4", content="another content", meta={"category": "test", "priority": 3}),
        ]

        document_store.write_documents(docs)

        # Verify documents are written
        assert document_store.count_documents() == 4

        # Search with embedding similar to first document
        query_embedding = [0.1, 0.2, 0.3]
        results = document_store.search(query_embedding, limit=100)

        assert len(results) == 4, f"Expected 2 results, got {len(results)}"
        assert results[0].id == "search1"  # Most similar should be first
        assert results[3].id == "search4"  # Document without embedding should be last

    def test_search_by_embedding_with_limit(self, document_store):
        docs = [
            Document(
                id="search1",
                content="similar content",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "test", "priority": 1},
                blob=ByteStream(data=b"binary_data", mime_type="application/octet-stream"),
                score=0.95,
            ),
            Document(
                id="search2",
                content="different content",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "other", "priority": 2},
                score=0.85,
            ),
            Document(
                id="search3",
                content="another content",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "test", "priority": 3},
            ),
            Document(id="search4", content="another content", meta={"category": "test", "priority": 3}),
        ]

        document_store.write_documents(docs)

        # Verify documents are written
        assert document_store.count_documents() == 4

        # Search with embedding similar to first document
        query_embedding = [0.1, 0.2, 0.3]
        results = document_store.search(query_embedding, limit=2)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert results[0].id == "search1"  # Most similar should be first
        assert results[1].id == "search3"  # Document without embedding should be last

    def test_search_by_embedding_with_category_filter(self, document_store):
        docs = [
            Document(
                id="search1",
                content="similar content",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "test", "priority": 1},
                blob=ByteStream(data=b"binary_data", mime_type="application/octet-stream"),
                score=0.95,
            ),
            Document(
                id="search2",
                content="different content",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "other", "priority": 2},
                score=0.85,
            ),
            Document(
                id="search3",
                content="another content",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "test2", "priority": 3},
            ),
            Document(id="search4", content="another content", meta={"category": "test3", "priority": 3}),
        ]

        document_store.write_documents(docs)

        # Verify documents are written
        assert document_store.count_documents() == 4

        # Search with embedding similar to first document
        query_embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "test"}]}
        results = document_store.search(query_embedding, filters, limit=2)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0].id == "search1"  # Most similar should be first

    def test_search_by_embedding_with_numeric_filter(self, document_store):
        docs = [
            Document(id="n1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"priority": 1, "score": 0.8}),
            Document(id="n2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"priority": 5, "score": 0.9}),
            Document(id="n3", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"priority": 10, "score": 0.7}),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": ">=", "value": 5}]}
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"n2", "n3"}
        # Results should be ordered by similarity to query
        assert results[0].id == "n2"  # Closer to query embedding

    def test_search_by_embedding_with_or_filter(self, document_store):
        docs = [
            Document(
                id="o1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"status": "active", "category": "news"}
            ),
            Document(
                id="o2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"status": "pending", "category": "sports"}
            ),
            Document(
                id="o3", content="doc 3", embedding=[0.9, 0.8, 0.7], meta={"status": "inactive", "category": "news"}
            ),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.status", "operator": "==", "value": "active"},
                {"field": "meta.category", "operator": "==", "value": "sports"},
            ],
        }
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"o1", "o2"}
        # Results should be ordered by similarity
        assert results[0].id == "o1"  # Closer to query embedding

    def test_search_by_embedding_with_in_filter(self, document_store):
        docs = [
            Document(id="i1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"status": "active"}),
            Document(id="i2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"status": "pending"}),
            Document(id="i3", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"status": "inactive"}),
            Document(id="i4", content="doc 4", embedding=[0.9, 0.8, 0.7], meta={"status": "archived"}),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "AND",
            "conditions": [{"field": "meta.status", "operator": "in", "value": ["active", "pending"]}],
        }
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"i1", "i2"}
        # Results should be ordered by similarity
        assert results[0].id == "i1"  # Closest to query

    def test_search_by_embedding_with_complex_nested_filter(self, document_store):
        docs = [
            Document(
                id="c1",
                content="doc 1",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "news", "priority": 1, "status": "active"},
            ),
            Document(
                id="c2",
                content="doc 2",
                embedding=[0.2, 0.3, 0.4],
                meta={"category": "news", "priority": 5, "status": "pending"},
            ),
            Document(
                id="c3",
                content="doc 3",
                embedding=[0.3, 0.4, 0.5],
                meta={"category": "sports", "priority": 3, "status": "active"},
            ),
            Document(
                id="c4",
                content="doc 4",
                embedding=[0.9, 0.8, 0.7],
                meta={"category": "sports", "priority": 8, "status": "inactive"},
            ),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.priority", "operator": ">", "value": 3},
                        {"field": "meta.status", "operator": "==", "value": "active"},
                    ],
                },
            ],
        }
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"c1", "c2"}
        # Results should be ordered by similarity
        assert results[0].id == "c1"  # Closest to query

    def test_search_by_embedding_with_not_equal_filter(self, document_store):
        docs = [
            Document(id="ne1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"category": "news"}),
            Document(id="ne2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"category": "sports"}),
            Document(id="ne3", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"category": "spam"}),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "!=", "value": "spam"}]}
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"ne1", "ne2"}
        # Results should be ordered by similarity
        assert results[0].id == "ne1"  # Closest to query

    def test_search_by_embedding_with_range_filter(self, document_store):
        docs = [
            Document(id="r1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"timestamp": 100, "priority": 1}),
            Document(id="r2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"timestamp": 800, "priority": 5}),
            Document(id="r3", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"timestamp": 900, "priority": 10}),
            Document(id="r4", content="doc 4", embedding=[0.9, 0.8, 0.7], meta={"timestamp": 300, "priority": 2}),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.timestamp", "operator": ">=", "value": 700},
                {"field": "meta.priority", "operator": "<=", "value": 8},
            ],
        }
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 1
        assert {doc.id for doc in results} == {"r2"}
        # Results should be ordered by similarity
        assert results[0].id == "r2"  # Only r2 matches: timestamp=800>=700 AND priority=5<=8

    def test_search_by_embedding_no_filter_matches(self, document_store):
        docs = [
            Document(id="nm1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"category": "news"}),
            Document(id="nm2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"category": "sports"}),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {
            "operator": "AND",
            "conditions": [{"field": "meta.category", "operator": "==", "value": "nonexistent"}],
        }
        results = document_store.search(query_embedding, filters, limit=10)

        assert len(results) == 0

    def test_filter_documents_by_category(self, document_store):
        docs = [
            Document(id="f1", content="doc 1", meta={"category": "news", "priority": 1}),
            Document(id="f2", content="doc 2", meta={"category": "sports", "priority": 2}),
            Document(id="f3", content="doc 3", meta={"category": "news", "priority": 3}),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]}
        results = document_store.filter_documents(filters)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"f1", "f3"}

    def test_filter_documents_by_numeric_range(self, document_store):
        docs = [
            Document(id="n1", content="doc 1", meta={"priority": 1, "score": 0.8}),
            Document(id="n2", content="doc 2", meta={"priority": 5, "score": 0.9}),
            Document(id="n3", content="doc 3", meta={"priority": 10, "score": 0.7}),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.priority", "operator": ">=", "value": 5},
                {"field": "meta.score", "operator": ">=", "value": 0.8},
            ],
        }
        results = document_store.filter_documents(filters)

        assert len(results) == 1
        assert results[0].id == "n2"

    def test_filter_documents_with_or_condition(self, document_store):
        docs = [
            Document(id="o1", content="doc 1", meta={"status": "active", "category": "news"}),
            Document(id="o2", content="doc 2", meta={"status": "pending", "category": "sports"}),
            Document(id="o3", content="doc 3", meta={"status": "inactive", "category": "news"}),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.status", "operator": "==", "value": "active"},
                {"field": "meta.category", "operator": "==", "value": "sports"},
            ],
        }
        results = document_store.filter_documents(filters)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"o1", "o2"}

    def test_filter_documents_with_in_operator(self, document_store):
        docs = [
            Document(id="i1", content="doc 1", meta={"status": "active"}),
            Document(id="i2", content="doc 2", meta={"status": "pending"}),
            Document(id="i3", content="doc 3", meta={"status": "inactive"}),
            Document(id="i4", content="doc 4", meta={"status": "archived"}),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [{"field": "meta.status", "operator": "in", "value": ["active", "pending"]}],
        }
        results = document_store.filter_documents(filters)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"i1", "i2"}

    def test_filter_documents_with_not_equal(self, document_store):
        docs = [
            Document(id="ne1", content="doc 1", meta={"category": "news"}),
            Document(id="ne2", content="doc 2", meta={"category": "sports"}),
            Document(id="ne3", content="doc 3", meta={"category": "spam"}),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "!=", "value": "spam"}]}
        results = document_store.filter_documents(filters)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"ne1", "ne2"}

    def test_filter_documents_no_matches(self, document_store):
        docs = [
            Document(id="nm1", content="doc 1", meta={"category": "news"}),
            Document(id="nm2", content="doc 2", meta={"category": "sports"}),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [{"field": "meta.category", "operator": "==", "value": "nonexistent"}],
        }
        results = document_store.filter_documents(filters)

        assert len(results) == 0

    def test_filter_documents_no_filters(self, document_store):
        docs = [
            Document(id="nf1", content="doc 1", meta={"category": "news"}),
            Document(id="nf2", content="doc 2", meta={"category": "sports"}),
        ]
        document_store.write_documents(docs)

        results = document_store.filter_documents(None)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"nf1", "nf2"}

    def test_filter_documents_complex_nested(self, document_store):
        docs = [
            Document(id="c1", content="doc 1", meta={"category": "news", "priority": 1, "status": "active"}),
            Document(id="c2", content="doc 2", meta={"category": "news", "priority": 5, "status": "pending"}),
            Document(id="c3", content="doc 3", meta={"category": "sports", "priority": 3, "status": "active"}),
            Document(id="c4", content="doc 4", meta={"category": "sports", "priority": 8, "status": "inactive"}),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.priority", "operator": ">", "value": 3},
                        {"field": "meta.status", "operator": "==", "value": "active"},
                    ],
                },
            ],
        }
        results = document_store.filter_documents(filters)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"c1", "c2"}

    def test_delete_all_documents(self, document_store):
        docs = [
            Document(id="del1", content="doc 1", embedding=[0.1, 0.2, 0.3]),
            Document(id="del2", content="doc 2", embedding=[0.4, 0.5, 0.6]),
            Document(id="del3", content="doc 3", embedding=[0.7, 0.8, 0.9]),
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_delete_all_documents_empty_store(self, document_store):
        assert document_store.count_documents() == 0

        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_similarity_scores_are_set_correctly(self, document_store):
        """Test that similarity scores are properly computed and set for all returned documents."""
        docs = [
            Document(id="sim1", content="identical vector", embedding=[1.0, 0.0, 0.0]),
            Document(id="sim2", content="similar vector", embedding=[0.9, 0.1, 0.0]),
            Document(id="sim3", content="different vector", embedding=[0.0, 1.0, 0.0]),
            Document(id="sim4", content="opposite vector", embedding=[-1.0, 0.0, 0.0]),
        ]

        document_store.write_documents(docs)

        # Search with query identical to first document
        query_embedding = [1.0, 0.0, 0.0]
        results = document_store.search(query_embedding, limit=10)

        # All documents should have similarity scores set
        assert len(results) == 4
        for doc in results:
            assert doc.score is not None, f"Document {doc.id} has no similarity score"
            assert isinstance(doc.score, float), f"Document {doc.id} score is not a float: {type(doc.score)}"

        # Results should be ordered by similarity (highest first for cosine similarity)
        assert results[0].id == "sim1"  # Identical vector should have highest score
        assert results[1].id == "sim2"  # Similar vector should be second

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

    def test_filter_by_meta_score(self, document_store):
        """Test filtering by user-provided meta.score values."""
        docs = [
            Document(id="ms1", content="doc 1", embedding=[0.1, 0.2, 0.3], meta={"score": 0.9, "category": "high"}),
            Document(id="ms2", content="doc 2", embedding=[0.2, 0.3, 0.4], meta={"score": 0.5, "category": "medium"}),
            Document(id="ms3", content="doc 3", embedding=[0.3, 0.4, 0.5], meta={"score": 0.2, "category": "low"}),
            Document(id="ms4", content="doc 4", embedding=[0.4, 0.5, 0.6], meta={"score": 0.8, "category": "high"}),
        ]
        document_store.write_documents(docs)

        # Filter by meta.score >= 0.7
        filters = {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": 0.7}]}
        results = document_store.filter_documents(filters)

        assert len(results) == 2
        assert {doc.id for doc in results} == {"ms1", "ms4"}

        # Verify meta.score values are preserved
        for doc in results:
            assert "score" in doc.meta
            assert doc.meta["score"] >= 0.7

    def test_search_with_meta_score_filter(self, document_store):
        """Test vector search combined with meta.score filtering."""
        docs = [
            Document(
                id="sms1", content="doc 1", embedding=[1.0, 0.0, 0.0], meta={"score": 0.9, "quality": "excellent"}
            ),
            Document(id="sms2", content="doc 2", embedding=[0.9, 0.1, 0.0], meta={"score": 0.3, "quality": "poor"}),
            Document(id="sms3", content="doc 3", embedding=[0.8, 0.2, 0.0], meta={"score": 0.8, "quality": "good"}),
            Document(
                id="sms4", content="doc 4", embedding=[0.0, 1.0, 0.0], meta={"score": 0.95, "quality": "excellent"}
            ),
        ]
        document_store.write_documents(docs)

        # Search with query similar to first document, but filter by meta.score
        query_embedding = [1.0, 0.0, 0.0]
        filters = {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": 0.7}]}
        results = document_store.search(query_embedding, filters, limit=10)

        # Should return documents with meta.score >= 0.7, ordered by vector similarity
        assert len(results) == 3
        result_ids = [doc.id for doc in results]
        assert set(result_ids) == {"sms1", "sms3", "sms4"}

        # Verify ordering by vector similarity (sms1 should be first as it's most similar)
        assert results[0].id == "sms1"

        # Verify both similarity scores and meta scores are preserved
        for doc in results:
            assert doc.score is not None  # Vector similarity score
            assert "score" in doc.meta  # User metadata score
            assert doc.meta["score"] >= 0.7
