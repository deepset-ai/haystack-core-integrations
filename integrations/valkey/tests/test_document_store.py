# ruff: noqa: S110
import struct

import pytest
from glide_shared.commands.server_modules.ft_options.ft_create_options import DistanceMetricType
from glide_shared.commands.server_modules.ft_options.ft_search_options import FtSearchOptions
from haystack.dataclasses import Document
from haystack.dataclasses.byte_stream import ByteStream
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack.utils import Secret

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


@pytest.mark.integration
class TestValkeyDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self):
        store = ValkeyDocumentStore(index_name="test_haystack_document", embedding_dim=3)
        yield store
        try:
            store._client.flushdb()
            store.close()
        except Exception:
            pass

    def test_write_documents(self, document_store):
        """Test default write_documents() behavior (OVERWRITE by default)."""
        docs = [Document(id="1", content="test doc 1")]
        assert document_store.write_documents(docs) == 1
        # Valkey overwrites by default
        assert document_store.write_documents(docs) == 1
        assert document_store.count_documents() == 1

    def test_write_documents_duplicate_fail(self, document_store):
        """Valkey only supports OVERWRITE policy, skip FAIL test."""
        pytest.skip("Valkey only supports DuplicatePolicy.OVERWRITE")

    def test_write_documents_duplicate_skip(self, document_store):
        """Valkey only supports OVERWRITE policy, skip SKIP test."""
        pytest.skip("Valkey only supports DuplicatePolicy.OVERWRITE")

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
        results = document_store._embedding_retrieval(query_embedding, limit=100)

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
        results = document_store._embedding_retrieval(query_embedding, limit=2)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=2)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, limit=10)

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
        results = document_store._embedding_retrieval(query_embedding, filters, limit=10)

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


class TestValkeyDocumentStoreStaticMethods:
    """Test static methods that were refactored from instance methods."""

    def test_parse_metric_valid_metrics(self):
        """Test _parse_metric static method with valid metrics."""
        assert ValkeyDocumentStore._parse_metric("l2") == DistanceMetricType.L2
        assert ValkeyDocumentStore._parse_metric("cosine") == DistanceMetricType.COSINE
        assert ValkeyDocumentStore._parse_metric("ip") == DistanceMetricType.IP

    def test_parse_metric_invalid_metric(self):
        """Test _parse_metric static method with invalid metric."""
        with pytest.raises(ValueError, match="Unsupported metric: invalid"):
            ValkeyDocumentStore._parse_metric("invalid")

    def test_to_float32_bytes(self):
        """Test _to_float32_bytes static method."""
        vec = [1.0, 2.5, -3.7]
        result = ValkeyDocumentStore._to_float32_bytes(vec)

        # Verify it's bytes
        assert isinstance(result, bytes)

        # Verify correct length (4 bytes per float)
        assert len(result) == len(vec) * 4

        # Verify correct values by unpacking
        unpacked = [struct.unpack("<f", result[i : i + 4])[0] for i in range(0, len(result), 4)]
        assert unpacked == pytest.approx(vec, rel=1e-6)

    def test_verify_node_list_valid(self):
        """Test _verify_node_list static method with valid node list."""
        # Should not raise any exception
        ValkeyDocumentStore._verify_node_list([("localhost", 6379)])
        ValkeyDocumentStore._verify_node_list([("host1", 6379), ("host2", 6380)])

    def test_verify_node_list_empty(self):
        """Test _verify_node_list static method with empty node list."""
        with pytest.raises(Exception, match="Node list is empty"):
            ValkeyDocumentStore._verify_node_list([])

        with pytest.raises(Exception, match="Node list is empty"):
            ValkeyDocumentStore._verify_node_list(None)

    def test_build_credentials_with_username_and_password(self):
        """Test _build_credentials static method with both username and password."""
        creds = ValkeyDocumentStore._build_credentials(Secret.from_token("user"), Secret.from_token("pass"))
        assert creds is not None
        assert creds.username == "user"
        assert creds.password == "pass"

    def test_build_credentials_with_username_only(self):
        """Test _build_credentials static method with username only."""
        # ServerCredentials requires password, so username-only should return None
        creds = ValkeyDocumentStore._build_credentials(Secret.from_token("user"), None)
        assert creds is None

    def test_build_credentials_with_password_only(self):
        """Test _build_credentials static method with password only."""
        creds = ValkeyDocumentStore._build_credentials(None, Secret.from_token("pass"))
        assert creds is not None
        # Username should default to None (ServerCredentials will use "default")
        assert creds.password == "pass"

    def test_build_credentials_with_neither(self):
        """Test _build_credentials static method with neither username nor password."""
        creds = ValkeyDocumentStore._build_credentials(None, None)
        assert creds is None

    def test_validate_documents_valid(self):
        """Test _validate_documents static method with valid documents."""
        docs = [
            Document(id="1", content="test"),
            Document(id="2", content="test2"),
        ]
        # Should not raise any exception
        ValkeyDocumentStore._validate_documents(docs)

    def test_validate_documents_invalid_type(self):
        """Test _validate_documents static method with invalid document type."""
        with pytest.raises(ValueError, match="expects a list of Documents"):
            ValkeyDocumentStore._validate_documents([Document(id="1", content="test"), "not_a_document"])

    def test_validate_policy_valid(self):
        """Test _validate_policy static method with valid policies."""
        # Should not raise any exception, but may log warnings
        ValkeyDocumentStore._validate_policy(DuplicatePolicy.NONE)
        ValkeyDocumentStore._validate_policy(DuplicatePolicy.OVERWRITE)

    def test_build_search_query_and_options_basic(self):
        """Test _build_search_query_and_options static method with basic parameters."""
        embedding = [0.1, 0.2, 0.3]
        filters = None
        limit = 10
        with_embedding = True

        query, options = ValkeyDocumentStore._build_search_query_and_options(
            embedding, filters, limit, with_embedding=with_embedding
        )

        assert isinstance(query, str)
        assert isinstance(options, FtSearchOptions)
        assert "KNN 10" in query
        assert "query_vector" in query
        assert "query_vector" in options.params

    def test_build_search_query_and_options_with_filters(self):
        """Test _build_search_query_and_options static method with filters."""
        embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]}
        limit = 5
        with_embedding = False

        query, options = ValkeyDocumentStore._build_search_query_and_options(
            embedding, filters, limit, with_embedding=with_embedding
        )

        assert "meta_category:{news}" in query
        assert "KNN 5" in query
        # Should not include vector field when with_embedding=False
        vector_fields = [
            field for field in options.return_fields if hasattr(field, "alias") and field.alias == "vector"
        ]
        assert len(vector_fields) == 0

    def test_build_search_query_and_options_with_embedding_return(self):
        """Test _build_search_query_and_options static method with embedding return."""
        embedding = [0.1, 0.2, 0.3]
        filters = None
        limit = 10

        _, options = ValkeyDocumentStore._build_search_query_and_options(embedding, filters, limit, with_embedding=True)

        # Should have vector field when with_embedding=True
        vector_fields = [
            field for field in options.return_fields if hasattr(field, "alias") and field.alias == "vector"
        ]
        assert len(vector_fields) == 1, "Should have exactly one vector field when with_embedding=True"

        # Should have more return fields when with_embedding=True vs False
        _, options_no_embed = ValkeyDocumentStore._build_search_query_and_options(
            embedding, filters, limit, with_embedding=False
        )

        vector_fields = [
            field for field in options_no_embed.return_fields if hasattr(field, "alias") and field.alias == "vector"
        ]
        assert len(vector_fields) == 0, "Should have no vector field with_embedding=False"

    def test_parse_documents_from_ft_empty_results(self):
        """Test _parse_documents_from_ft static method with empty results."""
        raw_results = [0, {}]  # Empty results format
        with_embedding = True

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=with_embedding)
        assert docs == []

    def test_parse_documents_from_ft_no_results(self):
        """Test _parse_documents_from_ft static method with no results."""
        raw_results = None
        with_embedding = True

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=with_embedding)
        assert docs == []

    def test_parse_documents_from_ft_with_embeddings(self):
        """Test _parse_documents_from_ft correctly parses embeddings when with_embedding=True."""
        raw_results = [
            2,
            {
                b"doc:1": {
                    b"payload": b'{"id": "1", "content": "test doc 1"}',
                    b"vector": b"[0.1, 0.2, 0.3]",
                    b"__vector_score": b"0.95",
                },
                b"doc:2": {
                    b"payload": b'{"id": "2", "content": "test doc 2"}',
                    b"vector": b"[0.4, 0.5, 0.6]",
                    b"__vector_score": b"0.85",
                },
            },
        ]

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=True)

        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[0].content == "test doc 1"
        assert docs[0].embedding == [0.1, 0.2, 0.3]
        assert docs[0].score == 0.95
        assert docs[1].id == "2"
        assert docs[1].embedding == [0.4, 0.5, 0.6]
        assert docs[1].score == 0.85

    def test_parse_documents_from_ft_without_embeddings(self):
        """Test _parse_documents_from_ft excludes embeddings when with_embedding=False."""
        raw_results = [
            2,
            {
                b"doc:1": {
                    b"payload": b'{"id": "1", "content": "test doc 1"}',
                    b"vector": b"[0.1, 0.2, 0.3]",
                    b"__vector_score": b"0.95",
                },
                b"doc:2": {
                    b"payload": b'{"id": "2", "content": "test doc 2"}',
                    b"vector": b"[0.4, 0.5, 0.6]",
                    b"__vector_score": b"0.85",
                },
            },
        ]

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=False)

        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[0].embedding is None
        assert docs[0].score == 0.95
        assert docs[1].id == "2"
        assert docs[1].embedding is None
        assert docs[1].score == 0.85

    def test_parse_documents_from_ft_filters_dummy_vectors(self):
        """Test _parse_documents_from_ft filters out dummy vectors."""
        raw_results = [
            1,
            {
                b"doc:1": {
                    b"payload": b'{"id": "1", "content": "test doc without embedding"}',
                    b"vector": b"[-10.0, -10.0, -10.0]",
                    b"__vector_score": b"0.5",
                },
            },
        ]

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=True)

        assert len(docs) == 1
        assert docs[0].id == "1"
        assert docs[0].embedding is None
        assert docs[0].score == 0.5

    def test_parse_documents_from_ft_missing_payload(self):
        """Test _parse_documents_from_ft skips documents without payload."""
        raw_results = [
            2,
            {
                b"doc:1": {
                    b"payload": b'{"id": "1", "content": "test doc 1"}',
                    b"__vector_score": b"0.95",
                },
                b"doc:2": {
                    b"vector": b"[0.4, 0.5, 0.6]",
                    b"__vector_score": b"0.85",
                },
            },
        ]

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=True)

        assert len(docs) == 1
        assert docs[0].id == "1"

    def test_parse_documents_from_ft_missing_score(self):
        """Test _parse_documents_from_ft handles missing similarity score."""
        raw_results = [
            1,
            {
                b"doc:1": {
                    b"payload": b'{"id": "1", "content": "test doc 1"}',
                    b"vector": b"[0.1, 0.2, 0.3]",
                },
            },
        ]

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=True)

        assert len(docs) == 1
        assert docs[0].id == "1"
        assert docs[0].score is None

    def test_class_constants_accessible(self):
        """Test that class constants are accessible and have correct values."""
        assert ValkeyDocumentStore._DUMMY_VALUE == -10.0
        assert "l2" in ValkeyDocumentStore._METRIC_MAP
        assert "cosine" in ValkeyDocumentStore._METRIC_MAP
        assert "ip" in ValkeyDocumentStore._METRIC_MAP
        assert ValkeyDocumentStore._METRIC_MAP["cosine"] == DistanceMetricType.COSINE

    def test_dummy_vector_consistency(self):
        """Test that dummy vector uses the class constant consistently."""
        store = ValkeyDocumentStore(embedding_dim=5)
        expected_dummy = [ValkeyDocumentStore._DUMMY_VALUE] * 5
        assert store._dummy_vector == expected_dummy

    def test_static_methods_dont_need_instance(self):
        """Test that static methods can be called without creating an instance."""
        # These should all work without instantiating ValkeyDocumentStore
        ValkeyDocumentStore._parse_metric("cosine")
        ValkeyDocumentStore._to_float32_bytes([1.0, 2.0])
        ValkeyDocumentStore._verify_node_list([("localhost", 6379)])
        ValkeyDocumentStore._build_credentials(Secret.from_token("user"), Secret.from_token("pass"))
        ValkeyDocumentStore._validate_documents([Document(id="1", content="test")])
        ValkeyDocumentStore._validate_policy(DuplicatePolicy.NONE)


class TestValkeyDocumentStoreConverters:
    def test_to_dict(self):
        document_store = ValkeyDocumentStore(
            nodes_list=[{"host": "localhost", "port": 6379}],
            cluster_mode=False,
            username=Secret.from_token("test_user"),
            password=Secret.from_token("test_pass"),
            request_timeout=30,
            index_name="test_index",
            distance_metric="cosine",
            embedding_dim=512,
        )

        result = document_store.to_dict()

        assert result["type"] == "haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore"
        assert result["init_parameters"]["nodes_list"] == [{"host": "localhost", "port": 6379}]
        assert result["init_parameters"]["cluster_mode"] is False
        assert result["init_parameters"]["request_timeout"] == 30
        assert result["init_parameters"]["index_name"] == "test_index"
        assert result["init_parameters"]["distance_metric"] == "cosine"
        assert result["init_parameters"]["embedding_dim"] == 512

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore",
            "init_parameters": {
                "nodes_list": [{"host": "localhost", "port": 6379}],
                "cluster_mode": True,
                "username": {"type": "env_var", "env_vars": ["TEST_USER"], "strict": False},
                "password": {"type": "env_var", "env_vars": ["TEST_PASS"], "strict": False},
                "request_timeout": 60,
                "index_name": "custom_index",
                "distance_metric": "l2",
                "embedding_dim": 768,
            },
        }

        document_store = ValkeyDocumentStore.from_dict(data)

        assert document_store._nodes_list == [{"host": "localhost", "port": 6379}]
        assert document_store._cluster_mode is True
        assert document_store._request_timeout == 60
        assert document_store._index_name == "custom_index"
        assert document_store._distance_metric.name.lower() == "l2"
        assert document_store._embedding_dim == 768
