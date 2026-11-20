"""Integration tests for CassandraDocumentStore.

These tests require a running Cassandra instance.
Run with: pytest tests/integration -v
"""

import pytest
from haystack import Document
from haystack.dataclasses import DuplicatePolicy

try:
    from haystack_integrations.document_stores.cassandra import CassandraDocumentStore
except ImportError:
    pytest.skip("cassandra-haystack not installed", allow_module_level=True)

# Skip if no Cassandra available
pytestmark = pytest.mark.integration


class TestCassandraDocumentStoreIntegration:
    """Integration tests requiring a live Cassandra instance."""

    @pytest.fixture(scope="class")
    def document_store(self):
        """Create a document store for testing."""
        # Use localhost Cassandra
        store = CassandraDocumentStore(
            contact_points=["127.0.0.1"],
            port=9042,
            keyspace="haystack_test",
            table="test_documents",
        )
        yield store
        # Cleanup
        store.delete_all_documents()
        store.close()

    def test_write_and_count(self, document_store):
        """Test writing documents and counting."""
        docs = [
            Document(content="Doc 1", meta={"type": "test"}),
            Document(content="Doc 2", meta={"type": "test"}),
            Document(content="Doc 3", meta={"type": "other"}),
        ]
        count = document_store.write_documents(docs)
        assert count == 3
        assert document_store.count_documents() == 3

    def test_write_with_ids(self, document_store):
        """Test writing documents with explicit IDs."""
        docs = [
            Document(id="custom-1", content="Doc 1"),
            Document(id="custom-2", content="Doc 2"),
        ]
        count = document_store.write_documents(docs)
        assert count == 2

    def test_write_duplicate_fail(self, document_store):
        """Test duplicate policy FAIL."""
        docs = [Document(id="dup-test", content="Original")]
        document_store.write_documents(docs)

        with pytest.raises(ValueError, match="already exists"):
            document_store.write_documents([Document(id="dup-test", content="Duplicate")])

    def test_write_duplicate_overwrite(self, document_store):
        """Test duplicate policy OVERWRITE."""
        docs = [Document(id="dup-test", content="Original")]
        document_store.write_documents(docs)

        count = document_store.write_documents(
            [Document(id="dup-test", content="Overwritten")],
            policy=DuplicatePolicy.OVERWRITE
        )
        assert count == 1

        # Verify overwrite
        results = document_store.filter_documents()
        matching = [d for d in results if d.id == "dup-test"]
        assert len(matching) == 1
        assert matching[0].content == "Overwritten"

    def test_write_duplicate_skip(self, document_store):
        """Test duplicate policy SKIP."""
        docs = [Document(id="dup-test", content="Original")]
        document_store.write_documents(docs)

        count = document_store.write_documents(
            [Document(id="dup-test", content="Skipped")],
            policy=DuplicatePolicy.SKIP
        )
        assert count == 0

        # Verify original remains
        results = document_store.filter_documents()
        matching = [d for d in results if d.id == "dup-test"]
        assert len(matching) == 1
        assert matching[0].content == "Original"

    def test_filter_documents(self, document_store):
        """Test filtering documents by metadata."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "value": 1}),
            Document(content="Doc 2", meta={"category": "B", "value": 2}),
            Document(content="Doc 3", meta={"category": "A", "value": 3}),
        ]
        document_store.write_documents(docs)

        # Filter by category
        results = document_store.filter_documents(filters={"category": "A"})
        assert len(results) == 2
        assert all(d.meta["category"] == "A" for d in results)

    def test_filter_documents_multiple(self, document_store):
        """Test filtering by multiple metadata fields."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "value": 1}),
            Document(content="Doc 2", meta={"category": "A", "value": 2}),
        ]
        document_store.write_documents(docs)

        results = document_store.filter_documents(filters={"category": "A", "value": 1})
        assert len(results) == 1
        assert results[0].content == "Doc 1"

    def test_delete_documents(self, document_store):
        """Test deleting documents by ID."""
        docs = [
            Document(id="del-1", content="Doc 1"),
            Document(id="del-2", content="Doc 2"),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_documents(["del-1"])
        assert document_store.count_documents() == 1

        remaining = document_store.filter_documents()
        assert len(remaining) == 1
        assert remaining[0].id == "del-2"

    def test_delete_all_documents(self, document_store):
        """Test deleting all documents."""
        docs = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_embedding_support(self, document_store):
        """Test documents with embeddings."""
        docs = [
            Document(
                content="Embedded doc",
                embedding=[0.1, 0.2, 0.3, 0.4],
                meta={"has_embedding": True}
            ),
        ]
        document_store.write_documents(docs)

        results = document_store.filter_documents(filters={"has_embedding": True})
        assert len(results) == 1
        assert results[0].embedding == [0.1, 0.2, 0.3, 0.4]

    def test_score_support(self, document_store):
        """Test documents with scores."""
        docs = [
            Document(content="Scored doc", score=0.95),
        ]
        document_store.write_documents(docs)

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].score == 0.95

    def test_empty_content(self, document_store):
        """Test document with empty content."""
        docs = [Document(content="", meta={"empty": True})]
        count = document_store.write_documents(docs)
        assert count == 1

        results = document_store.filter_documents(filters={"empty": True})
        assert len(results) == 1
        assert results[0].content == ""
