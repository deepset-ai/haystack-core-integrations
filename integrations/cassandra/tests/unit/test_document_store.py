"""Unit tests for CassandraDocumentStore."""

from unittest.mock import MagicMock, patch

from haystack import Document

# Mock cassandra imports
with patch.dict("sys.modules", {
    "cassandra": MagicMock(),
    "cassandra.auth": MagicMock(),
    "cassandra.cluster": MagicMock(),
    "cassandra.query": MagicMock(),
}):
    from haystack_integrations.document_stores.cassandra.document_store import CassandraDocumentStore


class TestCassandraDocumentStore:
    """Test CassandraDocumentStore initialization and serialization."""

    def test_init_defaults(self):
        """Test default initialization parameters."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()
            assert store._contact_points == ["127.0.0.1"]
            assert store._port == 9042
            assert store._keyspace == "haystack"
            assert store._table == "documents"
            assert store._ttl_seconds is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore(
                contact_points=["10.0.0.1", "10.0.0.2"],
                port=9043,
                username="user",
                password="pass",
                keyspace="custom_ks",
                table="custom_table",
                ttl_seconds=3600,
            )
            assert store._contact_points == ["10.0.0.1", "10.0.0.2"]
            assert store._port == 9043
            assert store._username == "user"
            assert store._password == "pass"
            assert store._keyspace == "custom_ks"
            assert store._table == "custom_table"
            assert store._ttl_seconds == 3600

    def test_init_with_secret(self):
        """Test initialization with Secret objects."""
        from haystack.utils import Secret
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore(
                username=Secret.from_token("secret_user"),
                password=Secret.from_token("secret_pass"),
            )
            assert store._username == "secret_user"
            assert store._password == "secret_pass"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore(
                contact_points=["10.0.0.1"],
                username="user",
                password="pass",
            )
            data = store.to_dict()
            assert data["type"] == "CassandraDocumentStore"
            assert data["init_parameters"]["contact_points"] == ["10.0.0.1"]
            assert data["init_parameters"]["username"] == "user"
            assert data["init_parameters"]["password"] == "pass"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "CassandraDocumentStore",
            "init_parameters": {
                "contact_points": ["10.0.0.1"],
                "port": 9042,
                "username": "user",
                "password": "pass",
                "keyspace": "test_ks",
                "table": "test_table",
                "ttl_seconds": 3600,
            },
        }
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore.from_dict(data)
            assert store._contact_points == ["10.0.0.1"]
            assert store._port == 9042
            assert store._username == "user"
            assert store._password == "pass"
            assert store._keyspace == "test_ks"
            assert store._table == "test_table"
            assert store._ttl_seconds == 3600

    def test_row_to_document(self):
        """Test row to document conversion."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()

            # Create mock row
            mock_row = MagicMock()
            mock_row.id = "doc1"
            mock_row.content = "Test content"
            mock_row.metadata = '{"key": "value"}'
            mock_row.embedding = [0.1, 0.2, 0.3]
            mock_row.score = 0.95

            doc = store._row_to_document(mock_row)
            assert doc.id == "doc1"
            assert doc.content == "Test content"
            assert doc.meta == {"key": "value"}
            assert doc.embedding == [0.1, 0.2, 0.3]
            assert doc.score == 0.95

    def test_row_to_document_none_metadata(self):
        """Test row to document with None metadata."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()

            mock_row = MagicMock()
            mock_row.id = "doc1"
            mock_row.content = "Test content"
            mock_row.metadata = None
            mock_row.embedding = None
            mock_row.score = None

            doc = store._row_to_document(mock_row)
            assert doc.meta == {}
            assert doc.embedding is None
            assert doc.score is None

    def test_row_to_document_invalid_json(self):
        """Test row to document with invalid JSON metadata."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()

            mock_row = MagicMock()
            mock_row.id = "doc1"
            mock_row.content = "Test content"
            mock_row.metadata = "invalid json{"
            mock_row.embedding = None
            mock_row.score = 0.0

            doc = store._row_to_document(mock_row)
            assert doc.meta == {}

    def test_matches_filters_empty(self):
        """Test filter matching with empty filters."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()
            doc = Document(content="test", meta={"key": "value"})
            assert store._matches_filters(doc, None) is True
            assert store._matches_filters(doc, {}) is True

    def test_matches_filters_match(self):
        """Test filter matching with matching filters."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()
            doc = Document(content="test", meta={"key": "value", "other": "data"})
            assert store._matches_filters(doc, {"key": "value"}) is True

    def test_matches_filters_no_match(self):
        """Test filter matching with non-matching filters."""
        with patch.object(CassandraDocumentStore, "_connect"), \
             patch.object(CassandraDocumentStore, "_ensure_schema"):
            store = CassandraDocumentStore()
            doc = Document(content="test", meta={"key": "value"})
            assert store._matches_filters(doc, {"key": "other"}) is False
            assert store._matches_filters(doc, {"missing": "value"}) is False
