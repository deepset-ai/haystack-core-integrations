# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IBM DB2 Document Store using live DB2 instance."""

import asyncio
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.ibm import Db2ConnectionConfig, Db2DocumentStore


# DB2 connection configuration for enterprise DB2 instance
DB2_CONFIG = Db2ConnectionConfig(
    database="TESTDB",
    hostname="HOST",
    port=50000,
    username="USER",
    password="PASS",
    protocol="TCPIP",
)


@pytest.fixture
def document_store():
    """Create a fresh document store for each test."""
    store = Db2DocumentStore(
        connection_config=DB2_CONFIG,
        table_name="test_haystack_docs",
        embedding_dim=384,
        distance_metric="COSINE",
        recreate_table=True,
    )
    yield store
    # Cleanup after test
    try:
        conn = store._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {store.table_name}")
            conn.commit()
    except Exception:
        pass


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="This is the first document about Python programming.",
            meta={"category": "programming", "language": "python", "level": "beginner", "rating": 5},
            embedding=[0.1] * 384,
        ),
        Document(
            id="doc2",
            content="This is the second document about Java development.",
            meta={"category": "programming", "language": "java", "level": "intermediate", "rating": 4},
            embedding=[0.2] * 384,
        ),
        Document(
            id="doc3",
            content="This is the third document about data science.",
            meta={"category": "data-science", "language": "python", "level": "advanced", "rating": 5},
            embedding=[0.3] * 384,
        ),
    ]


@pytest.mark.integration
class TestDb2DocumentStoreBasicOperations:
    """Test basic CRUD operations."""

    def test_count_documents_empty(self, document_store):
        """Test counting documents in empty store."""
        count = document_store.count_documents()
        assert count == 0

    def test_write_documents_basic(self, document_store, sample_documents):
        """Test writing documents to the store."""
        written = document_store.write_documents(sample_documents)
        assert written == 3
        assert document_store.count_documents() == 3

    def test_write_documents_empty_list(self, document_store):
        """Test writing empty list of documents."""
        written = document_store.write_documents([])
        assert written == 0
        assert document_store.count_documents() == 0

    def test_filter_documents_all(self, document_store, sample_documents):
        """Test retrieving all documents without filters."""
        document_store.write_documents(sample_documents)
        docs = document_store.filter_documents()
        assert len(docs) == 3
        assert {doc.id for doc in docs} == {"doc1", "doc2", "doc3"}

    def test_filter_documents_by_id(self, document_store, sample_documents):
        """Test filtering documents by ID."""
        document_store.write_documents(sample_documents)
        
        # Filter for specific document
        filters = {
            "operator": "==",
            "field": "id",
            "value": "doc1"
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc1"
        assert docs[0].content == "This is the first document about Python programming."

    def test_delete_documents(self, document_store, sample_documents):
        """Test deleting documents by IDs."""
        document_store.write_documents(sample_documents)
        assert document_store.count_documents() == 3
        
        document_store.delete_documents(["doc1", "doc3"])
        assert document_store.count_documents() == 1
        
        remaining = document_store.filter_documents()
        assert len(remaining) == 1
        assert remaining[0].id == "doc2"

    def test_delete_documents_empty_list(self, document_store, sample_documents):
        """Test deleting with empty list."""
        document_store.write_documents(sample_documents)
        document_store.delete_documents([])
        assert document_store.count_documents() == 3

    def test_delete_nonexistent_documents(self, document_store, sample_documents):
        """Test deleting non-existent documents."""
        document_store.write_documents(sample_documents)
        document_store.delete_documents(["nonexistent1", "nonexistent2"])
        assert document_store.count_documents() == 3


@pytest.mark.integration
class TestDb2DocumentStoreDuplicatePolicies:
    """Test duplicate handling policies."""

    def test_duplicate_policy_none(self, document_store, sample_documents):
        """Test NONE policy - allows duplicates (should fail at DB level)."""
        document_store.write_documents(sample_documents)
        
        # Try to insert duplicate - should fail due to primary key constraint
        with pytest.raises(Exception):  # DB will raise an error
            document_store.write_documents([sample_documents[0]], policy=DuplicatePolicy.NONE)

    def test_duplicate_policy_fail(self, document_store, sample_documents):
        """Test FAIL policy - raises DuplicateDocumentError."""
        document_store.write_documents(sample_documents)
        
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([sample_documents[0]], policy=DuplicatePolicy.FAIL)

    def test_duplicate_policy_skip(self, document_store, sample_documents):
        """Test SKIP policy - skips duplicates."""
        document_store.write_documents(sample_documents)
        
        # Try to insert duplicates - should skip them
        duplicate_doc = Document(
            id="doc1",
            content="Updated content",
            meta={"updated": True},
            embedding=[0.9] * 384,
        )
        written = document_store.write_documents([duplicate_doc], policy=DuplicatePolicy.SKIP)
        assert written == 0  # No documents written
        
        # Original document should remain unchanged
        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "doc1"})
        assert len(docs) == 1
        assert docs[0].content == "This is the first document about Python programming."
        assert docs[0].meta.get("updated") is None

    def test_duplicate_policy_overwrite(self, document_store, sample_documents):
        """Test OVERWRITE policy - updates existing documents."""
        document_store.write_documents(sample_documents)
        
        # Update existing document
        updated_doc = Document(
            id="doc1",
            content="Updated content for doc1",
            meta={"category": "programming", "updated": True, "rating": 10},
            embedding=[0.9] * 384,
        )
        written = document_store.write_documents([updated_doc], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        
        # Verify document was updated
        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "doc1"})
        assert len(docs) == 1
        assert docs[0].content == "Updated content for doc1"
        assert docs[0].meta["updated"] is True
        assert docs[0].meta["rating"] == 10

@pytest.mark.integration
class TestDb2DocumentStorePureSQLFiltering:
    """Test pure SQL filtering approach (similar to PgVector)."""

    def test_filter_by_metadata_string(self, document_store, sample_documents):
        """Test filtering by metadata string field using pure SQL."""
        document_store.write_documents(sample_documents)
        
        # Filter by language metadata
        filters = {
            "operator": "==",
            "field": "meta.language",
            "value": "python"
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_by_metadata_integer(self, document_store, sample_documents):
        """Test filtering by metadata integer field using pure SQL."""
        document_store.write_documents(sample_documents)
        
        # Filter by rating metadata
        filters = {
            "operator": "==",
            "field": "meta.rating",
            "value": "5"  # Note: stored as string in JSON
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_by_metadata_comparison(self, document_store, sample_documents):
        """Test comparison operators on metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter by rating > 4
        filters = {
            "operator": ">",
            "field": "meta.rating",
            "value": "4"
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_with_and_operator(self, document_store, sample_documents):
        """Test AND logical operator with metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter: language == "python" AND rating == 5
        filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
                {"operator": "==", "field": "meta.rating", "value": "5"}
            ]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_with_or_operator(self, document_store, sample_documents):
        """Test OR logical operator with metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter: language == "java" OR category == "data-science"
        filters = {
            "operator": "OR",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "java"},
                {"operator": "==", "field": "meta.category", "value": "data-science"}
            ]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc2", "doc3"}

    def test_filter_with_not_operator(self, document_store, sample_documents):
        """Test NOT logical operator with metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter: NOT (language == "java")
        filters = {
            "operator": "NOT",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "java"}
            ]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_with_in_operator(self, document_store, sample_documents):
        """Test IN operator with metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter: level IN ["beginner", "advanced"]
        filters = {
            "operator": "in",
            "field": "meta.level",
            "value": ["beginner", "advanced"]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_with_not_in_operator(self, document_store, sample_documents):
        """Test NOT IN operator with metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter: level NOT IN ["intermediate"]
        filters = {
            "operator": "not in",
            "field": "meta.level",
            "value": ["intermediate"]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_complex_nested_conditions(self, document_store, sample_documents):
        """Test complex nested logical conditions."""
        document_store.write_documents(sample_documents)
        
        # Filter: (language == "python" AND rating == 5) OR category == "data-science"
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"operator": "==", "field": "meta.language", "value": "python"},
                        {"operator": "==", "field": "meta.rating", "value": "5"}
                    ]
                },
                {"operator": "==", "field": "meta.category", "value": "data-science"}
            ]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_mixed_direct_and_metadata_fields(self, document_store, sample_documents):
        """Test filtering with both direct columns and metadata fields."""
        document_store.write_documents(sample_documents)
        
        # Filter: id == "doc1" AND language == "python"
        filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "id", "value": "doc1"},
                {"operator": "==", "field": "meta.language", "value": "python"}
            ]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc1"

    def test_filter_no_results(self, document_store, sample_documents):
        """Test filter that returns no results."""
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "==",
            "field": "meta.language",
            "value": "nonexistent"
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 0

    @pytest.mark.asyncio
    async def test_filter_documents_async_with_metadata(self, document_store, sample_documents):
        """Test async filtering with metadata fields."""
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "==",
            "field": "meta.language",
            "value": "python"
        }
        docs = await document_store.filter_documents_async(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_without_meta_prefix(self, document_store, sample_documents):
        """Test filtering metadata fields without 'meta.' prefix."""
        document_store.write_documents(sample_documents)
        
        # Filter by language without "meta." prefix
        filters = {
            "operator": "==",
            "field": "language",  # No "meta." prefix
            "value": "python"
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}
        
        # Total count should remain the same
        assert document_store.count_documents() == 3

    def test_duplicate_policy_overwrite_mixed(self, document_store, sample_documents):
        """Test OVERWRITE policy with mix of new and existing documents."""
        document_store.write_documents([sample_documents[0]])
        
        # Mix of existing and new documents
        mixed_docs = [
            Document(id="doc1", content="Updated doc1", meta={"updated": True}, embedding=[0.9] * 384),
            Document(id="doc4", content="New doc4", meta={"new": True}, embedding=[0.4] * 384),
        ]
        written = document_store.write_documents(mixed_docs, policy=DuplicatePolicy.OVERWRITE)
        assert written == 2
        assert document_store.count_documents() == 2


@pytest.mark.integration
class TestDb2DocumentStoreFiltering:
    """Test filtering operations with various operators."""

    def test_filter_equality(self, document_store, sample_documents):
        """Test equality filter."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_inequality(self, document_store, sample_documents):
        """Test inequality filter."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "!=", "field": "meta.language", "value": "python"}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc2"

    def test_filter_greater_than(self, document_store, sample_documents):
        """Test greater than filter."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": ">", "field": "meta.rating", "value": 4}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_greater_than_or_equal(self, document_store, sample_documents):
        """Test greater than or equal filter."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": ">=", "field": "meta.rating", "value": 5}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_less_than(self, document_store, sample_documents):
        """Test less than filter."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "<", "field": "meta.rating", "value": 5}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc2"

    def test_filter_less_than_or_equal(self, document_store, sample_documents):
        """Test less than or equal filter."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "<=", "field": "meta.rating", "value": 4}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc2"

    def test_filter_in(self, document_store, sample_documents):
        """Test IN operator."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "in", "field": "meta.language", "value": ["python", "java"]}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 3

    def test_filter_not_in(self, document_store, sample_documents):
        """Test NOT IN operator."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "not in", "field": "meta.language", "value": ["java"]}
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    def test_filter_and(self, document_store, sample_documents):
        """Test AND logical operator."""
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
                {"operator": "==", "field": "meta.level", "value": "beginner"},
            ],
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc1"

    def test_filter_or(self, document_store, sample_documents):
        """Test OR logical operator."""
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "OR",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "java"},
                {"operator": "==", "field": "meta.level", "value": "advanced"},
            ],
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc2", "doc3"}

    def test_filter_not(self, document_store, sample_documents):
        """Test NOT logical operator."""
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "NOT",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
            ],
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc2"

    def test_filter_complex_nested(self, document_store, sample_documents):
        """Test complex nested filters."""
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "AND",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [
                        {"operator": "==", "field": "meta.language", "value": "python"},
                        {"operator": "==", "field": "meta.language", "value": "java"},
                    ],
                },
                {"operator": ">=", "field": "meta.rating", "value": 4},
            ],
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 3


@pytest.mark.integration
class TestDb2DocumentStoreAsync:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_count_documents_async(self, document_store, sample_documents):
        """Test async document counting."""
        document_store.write_documents(sample_documents)
        count = await document_store.count_documents_async()
        assert count == 3

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store, sample_documents):
        """Test async document writing."""
        written = await document_store.write_documents_async(sample_documents)
        assert written == 3
        assert document_store.count_documents() == 3

    @pytest.mark.asyncio
    async def test_filter_documents_async(self, document_store, sample_documents):
        """Test async document filtering."""
        document_store.write_documents(sample_documents)
        
        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        docs = await document_store.filter_documents_async(filters)
        assert len(docs) == 2
        assert {doc.id for doc in docs} == {"doc1", "doc3"}

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, document_store, sample_documents):
        """Test async document deletion."""
        document_store.write_documents(sample_documents)
        await document_store.delete_documents_async(["doc1", "doc3"])
        
        count = await document_store.count_documents_async()
        assert count == 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, document_store, sample_documents):
        """Test concurrent async operations."""
        # Write documents first
        document_store.write_documents(sample_documents)
        
        # Run multiple operations concurrently
        results = await asyncio.gather(
            document_store.count_documents_async(),
            document_store.filter_documents_async({"operator": "==", "field": "meta.language", "value": "python"}),
            document_store.filter_documents_async({"operator": "==", "field": "meta.language", "value": "java"}),
        )
        
        count, python_docs, java_docs = results
        assert count == 3
        assert len(python_docs) == 2
        assert len(java_docs) == 1


@pytest.mark.integration
class TestDb2DocumentStoreSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self, document_store):
        """Test serializing document store to dictionary."""
        data = document_store.to_dict()
        
        assert "type" in data
        assert data["type"] == "haystack_integrations.document_stores.ibm.document_store.Db2DocumentStore"
        assert "init_parameters" in data
        
        init_params = data["init_parameters"]
        assert "connection_config" in init_params
        assert init_params["table_name"] == "test_haystack_docs"
        assert init_params["embedding_dim"] == 384
        assert init_params["distance_metric"] == "COSINE"
        
        conn_config = init_params["connection_config"]
        # DB2 may return database name in uppercase
        assert conn_config["database"].upper() == "TESTDB"
        assert conn_config["hostname"] in ("localhost", "Geetika-5y420-x86.dev.fyre.ibm.com")
        assert conn_config["port"] == 50000

    def test_from_dict(self, document_store):
        """Test deserializing document store from dictionary."""
        data = document_store.to_dict()
        
        # Create new instance from dict
        new_store = Db2DocumentStore.from_dict(data)
        
        assert new_store.table_name == document_store.table_name
        assert new_store.embedding_dim == document_store.embedding_dim
        assert new_store.distance_metric == document_store.distance_metric
        assert new_store.connection_config.database == document_store.connection_config.database
        assert new_store.connection_config.hostname == document_store.connection_config.hostname

    def test_roundtrip_serialization(self, document_store, sample_documents):
        """Test full roundtrip serialization with data."""
        # Write some documents
        document_store.write_documents(sample_documents)
        
        # Serialize
        data = document_store.to_dict()
        
        # Deserialize
        new_store = Db2DocumentStore.from_dict(data)
        
        # Verify data is accessible
        count = new_store.count_documents()
        assert count == 3
        
        docs = new_store.filter_documents()
        assert len(docs) == 3


@pytest.mark.integration
class TestDb2DocumentStoreEdgeCases:
    """Test edge cases and error handling."""

    def test_document_without_embedding(self, document_store):
        """Test storing document without embedding."""
        doc = Document(id="no_emb", content="Document without embedding", meta={"test": True})
        document_store.write_documents([doc])
        
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_emb"})
        assert len(retrieved) == 1
        assert retrieved[0].embedding is None

    def test_document_without_content(self, document_store):
        """Test storing document without content."""
        doc = Document(id="no_content", content=None, meta={"test": True}, embedding=[0.1] * 384)
        document_store.write_documents([doc])
        
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_content"})
        assert len(retrieved) == 1
        assert retrieved[0].content is None

    def test_document_without_meta(self, document_store):
        """Test storing document without metadata."""
        doc = Document(id="no_meta", content="Document without metadata", embedding=[0.1] * 384)
        document_store.write_documents([doc])
        
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_meta"})
        assert len(retrieved) == 1
        assert retrieved[0].meta == {}

    def test_large_document_content(self, document_store):
        """Test storing document with large content."""
        large_content = "A" * 100000  # 100KB of text
        doc = Document(id="large_doc", content=large_content, embedding=[0.1] * 384)
        document_store.write_documents([doc])
        
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "large_doc"})
        assert len(retrieved) == 1
        assert len(retrieved[0].content) == 100000

    def test_complex_metadata(self, document_store):
        """Test storing document with complex nested metadata."""
        doc = Document(
            id="complex_meta",
            content="Document with complex metadata",
            meta={
                "nested": {"level1": {"level2": {"level3": "deep"}}},
                "list": [1, 2, 3, "four"],
                "mixed": {"numbers": [1, 2, 3], "strings": ["a", "b", "c"]},
            },
            embedding=[0.1] * 384,
        )
        document_store.write_documents([doc])
        
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "complex_meta"})
        assert len(retrieved) == 1
        assert retrieved[0].meta["nested"]["level1"]["level2"]["level3"] == "deep"
        assert retrieved[0].meta["list"] == [1, 2, 3, "four"]

    def test_invalid_filter_operator(self, document_store, sample_documents):
        """Test invalid filter operator."""
        document_store.write_documents(sample_documents)
        
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            document_store.filter_documents({"operator": "INVALID", "field": "meta.language", "value": "python"})

    def test_filter_missing_operator(self, document_store, sample_documents):
        """Test filter without operator."""
        document_store.write_documents(sample_documents)
        
        with pytest.raises(ValueError, match="must include an 'operator' key"):
            document_store.filter_documents({"field": "meta.language", "value": "python"})

    def test_filter_missing_field(self, document_store, sample_documents):
        """Test comparison filter without field."""
        document_store.write_documents(sample_documents)
        
        with pytest.raises(ValueError, match="must include a 'field' key"):
            document_store.filter_documents({"operator": "==", "value": "python"})


@pytest.mark.integration
class TestDb2DocumentStoreConnection:
    """Test connection handling."""

    def test_connection_reuse(self, document_store, sample_documents):
        """Test that connection is reused across operations."""
        # Perform multiple operations
        document_store.write_documents(sample_documents)
        count1 = document_store.count_documents()
        docs = document_store.filter_documents()
        count2 = document_store.count_documents()
        
        assert count1 == count2 == 3
        assert len(docs) == 3
        
        # Connection should be the same instance
        conn1 = document_store._get_connection()
        conn2 = document_store._get_connection()
        assert conn1 is conn2

    def test_multiple_document_stores_same_table(self):
        """Test multiple document store instances accessing same table."""
        # Create first store and write data
        store1 = Db2DocumentStore(
            connection_config=DB2_CONFIG,
            table_name="shared_table",
            embedding_dim=384,
            recreate_table=True,
        )
        
        doc1 = Document(id="shared1", content="First document", embedding=[0.1] * 384)
        store1.write_documents([doc1])
        
        # Create second store accessing same table
        store2 = Db2DocumentStore(
            connection_config=DB2_CONFIG,
            table_name="shared_table",
            embedding_dim=384,
            recreate_table=False,
        )
        
        # Both should see the same data
        assert store1.count_documents() == 1
        assert store2.count_documents() == 1
        
        # Write from second store
        doc2 = Document(id="shared2", content="Second document", embedding=[0.2] * 384)
        store2.write_documents([doc2])
        
        # Both should see updated data
        assert store1.count_documents() == 2
        assert store2.count_documents() == 2
        
        # Cleanup
        try:
            conn = store1._get_connection()
            with conn.cursor() as cur:
                cur.execute("DROP TABLE shared_table")
                conn.commit()
        except Exception:
            pass

# Made with Bob
