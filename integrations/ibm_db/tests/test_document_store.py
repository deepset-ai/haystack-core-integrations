# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IBM DB2 Document Store using Haystack mixin tests."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    FilterDocumentsTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
)

from haystack_integrations.document_stores.ibm_db import Db2ConnectionConfig, Db2DocumentStore
from haystack_integrations.document_stores.ibm_db.document_store import _parse_embedding, _row_to_document

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


def _generate_self_signed_cert_pem() -> bytes:
    """
    Generate a self-signed SSL certificate for testing.

    :return: Certificate in PEM format as bytes
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        msg = "cryptography library is required to generate SSL certificates"
        raise ImportError(msg)

    # Generate private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    # Create certificate
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Test"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Test"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # Return certificate in PEM format
    return cert.public_bytes(serialization.Encoding.PEM)


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    WriteDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Test Db2DocumentStore using Haystack's standard mixin tests.

    This class inherits from Haystack's mixin test classes which provide
    standardized tests for document store implementations.
    """

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal, ignoring order.

        DB2 returns documents ordered by ID, but the expected list from Python
        is in insertion order. We sort both lists by ID before comparing.
        """
        # Sort both lists by document ID for consistent comparison
        received_sorted = sorted(received, key=lambda d: d.id)
        expected_sorted = sorted(expected, key=lambda d: d.id)

        # Check lengths first
        assert len(received_sorted) == len(expected_sorted), (
            f"Different number of documents: {len(received_sorted)} vs {len(expected_sorted)}"
        )

        # Compare each document
        for i, (rec, exp) in enumerate(zip(received_sorted, expected_sorted, strict=True)):
            assert rec.id == exp.id, f"Document {i}: IDs don't match: {rec.id} vs {exp.id}"
            assert rec.content == exp.content, f"Document {i} ({rec.id}): Content doesn't match"
            assert rec.meta == exp.meta, f"Document {i} ({rec.id}): Meta doesn't match: {rec.meta} vs {exp.meta}"

            # Handle embedding comparison with floating point tolerance
            if rec.embedding is None and exp.embedding is None:
                continue
            elif rec.embedding is None or exp.embedding is None:
                msg = f"Document {i} ({rec.id}): One embedding is None, the other is not"
                raise AssertionError(msg)
            else:
                assert len(rec.embedding) == len(exp.embedding), (
                    f"Document {i} ({rec.id}): Embedding lengths don't match"
                )
                # Compare embeddings with tolerance for floating point precision
                for j, (r_val, e_val) in enumerate(zip(rec.embedding, exp.embedding, strict=True)):
                    if not math.isclose(r_val, e_val, rel_tol=1e-6, abs_tol=1e-9):
                        msg = f"Document {i} ({rec.id}): Embedding value {j} doesn't match: {r_val} vs {e_val}"
                        raise AssertionError(msg)

    def test_write_documents(self, document_store: Db2DocumentStore):
        """Test basic write with duplicate handling - default policy is NONE."""
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        # Default policy is NONE — a second write of the same doc raises DuplicateDocumentError
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc])

    def test_to_dict(self, document_store: Db2DocumentStore):
        """Test serializing document store to dictionary."""
        data = document_store.to_dict()

        assert "type" in data
        assert data["type"] == "haystack_integrations.document_stores.ibm_db.document_store.Db2DocumentStore"
        assert "init_parameters" in data

        init_params = data["init_parameters"]
        assert "connection_config" in init_params
        assert init_params["embedding_dim"] == 768
        assert init_params["distance_metric"] == "COSINE"

    def test_from_dict(self, document_store: Db2DocumentStore):
        """Test deserializing document store from dictionary."""
        data = document_store.to_dict()

        # Create new instance from dict
        new_store = Db2DocumentStore.from_dict(data)

        assert new_store.table_name == document_store.table_name
        assert new_store.embedding_dim == document_store.embedding_dim
        assert new_store.distance_metric == document_store.distance_metric
        assert new_store.connection_config.database == document_store.connection_config.database
        assert new_store.connection_config.hostname == document_store.connection_config.hostname

    def test_connection_reuse(self, document_store: Db2DocumentStore):
        """Test that connection is reused across operations."""
        docs = [Document(id="1", content="test", embedding=[0.1] * 768)]

        # Perform multiple operations
        document_store.write_documents(docs)
        count1 = document_store.count_documents()
        retrieved = document_store.filter_documents()
        count2 = document_store.count_documents()

        assert count1 == count2 == 1
        assert len(retrieved) == 1

        # Connection should be the same instance
        conn1 = document_store._get_connection()
        conn2 = document_store._get_connection()
        assert conn1 is conn2

    def test_document_without_embedding(self, document_store: Db2DocumentStore):
        """Test storing document without embedding."""
        doc = Document(id="no_emb", content="Document without embedding", meta={"test": True})
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_emb"})
        assert len(retrieved) == 1
        assert retrieved[0].embedding is None

    def test_document_without_content(self, document_store: Db2DocumentStore):
        """Test storing document without content."""
        doc = Document(id="no_content", content=None, meta={"test": True}, embedding=[0.1] * 768)
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_content"})
        assert len(retrieved) == 1
        assert retrieved[0].content is None

    def test_complex_metadata(self, document_store: Db2DocumentStore):
        """Test storing document with complex nested metadata."""
        doc = Document(
            id="complex_meta",
            content="Document with complex metadata",
            meta={
                "nested": {"level1": {"level2": {"level3": "deep"}}},
                "list": [1, 2, 3, "four"],
                "mixed": {"numbers": [1, 2, 3], "strings": ["a", "b", "c"]},
            },
            embedding=[0.1] * 768,
        )
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "complex_meta"})
        assert len(retrieved) == 1
        assert retrieved[0].meta["nested"]["level1"]["level2"]["level3"] == "deep"
        assert retrieved[0].meta["list"] == [1, 2, 3, "four"]


class TestDb2DocumentStoreUnit:
    """Unit tests for Db2DocumentStore that don't require a database."""

    def test_to_row_with_none_metadata(self, document_store):
        """Test _to_row with None metadata."""
        doc = Document(id="1", content="test", meta=None, embedding=[0.1] * 768)
        row = document_store._to_row(doc)

        assert row[0] == "1"  # id
        assert row[1] == "test"  # content
        assert row[2] == "{}"  # meta should be empty JSON object
        assert row[3] is not None  # embedding

    def test_to_row_with_none_embedding(self, document_store):
        """Test _to_row with None embedding."""
        doc = Document(id="1", content="test", meta={"key": "value"}, embedding=None)
        row = document_store._to_row(doc)

        assert row[0] == "1"  # id
        assert row[1] == "test"  # content
        assert '"key"' in row[2]  # meta JSON
        assert row[3] is None  # embedding should be None

    def test_build_where_clause_empty_filters(self, document_store):
        """Test _build_where_clause with empty filters."""
        where_clause, params = document_store._build_where_clause({})
        assert where_clause == ""
        assert params == []

    def test_write_documents_invalid_type(self, document_store):
        """Test write_documents with invalid type."""
        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            document_store.write_documents("not a list")

    def test_write_documents_invalid_document_type(self, document_store):
        """Test write_documents with invalid document type in list."""
        with pytest.raises(ValueError, match="Expected Document objects"):
            document_store.write_documents([{"id": "1", "content": "test"}])

    def test_write_documents_unsupported_policy(self, document_store):
        """Test write_documents with unsupported duplicate policy."""
        doc = Document(id="1", content="test", embedding=[0.1] * 768)

        # Create a mock unsupported policy
        class UnsupportedPolicy:
            pass

        with pytest.raises(ValueError, match="Unsupported duplicate policy"):
            document_store.write_documents([doc], policy=UnsupportedPolicy())

    def test_row_to_document_with_none_values(self):
        """Test _row_to_document with None values."""
        # Test with None content and meta
        row = ("doc_id", None, None, None)
        doc = _row_to_document(row)

        assert doc.id == "doc_id"
        assert doc.content is None
        assert doc.meta == {}
        assert doc.embedding is None

    def test_row_to_document_with_valid_embedding(self):
        """Test _row_to_document with valid embedding."""
        # Test with valid embedding (as tuple/list)
        embedding_data = [0.1, 0.2, 0.3]
        row = ("doc_id", "content", '{"key": "value"}', embedding_data)
        doc = _row_to_document(row)

        assert doc.id == "doc_id"
        assert doc.content == "content"
        assert doc.meta == {"key": "value"}
        assert doc.embedding == [0.1, 0.2, 0.3]


class TestDb2DocumentStoreUtils:
    """Unit tests for the pure utility/helper methods that don't require a database."""

    @pytest.fixture
    def store(self):
        """A Db2DocumentStore built without connecting to a database."""
        config = Db2ConnectionConfig(database="db", hostname="host", username="u", password="p")
        with patch.object(Db2DocumentStore, "_ensure_table_exists", return_value=None):
            return Db2DocumentStore(
                connection_config=config,
                table_name="unit_docs",
                embedding_dim=4,
                distance_metric="COSINE",
            )

    # --- _parse_embedding ---

    def test_parse_embedding_none_returns_none(self):
        assert _parse_embedding(None) is None

    def test_parse_embedding_list_of_ints_converted_to_floats(self):
        assert _parse_embedding([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_parse_embedding_json_string_list(self):
        assert _parse_embedding("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]

    def test_parse_embedding_tuple_via_iterable_fallback(self):
        assert _parse_embedding((0.5, 1.5)) == [0.5, 1.5]

    def test_parse_embedding_non_numeric_string_returns_none(self):
        # json.loads fails, iterable fallback over characters raises ValueError -> None
        assert _parse_embedding("not-a-vector") is None

    def test_parse_embedding_json_string_non_list_returns_none(self):
        # Valid JSON but not a list -> iterable fallback over its characters raises -> None
        assert _parse_embedding('{"a": 1}') is None

    def test_parse_embedding_non_iterable_returns_none(self):
        assert _parse_embedding(object()) is None

    # --- _validate_embedding ---

    def test_validate_embedding_none_allowed(self, store):
        # Should not raise
        store._validate_embedding(None, allow_none=True)

    def test_validate_embedding_none_not_allowed_raises_value_error(self, store):
        with pytest.raises(ValueError, match="cannot be None"):
            store._validate_embedding(None, allow_none=False)

    def test_validate_embedding_non_list_raises_type_error(self, store):
        with pytest.raises(TypeError, match="must be a list"):
            store._validate_embedding("not a list")

    def test_validate_embedding_empty_list_raises_value_error(self, store):
        with pytest.raises(ValueError, match="cannot be empty"):
            store._validate_embedding([])

    def test_validate_embedding_non_numeric_values_raise_type_error(self, store):
        with pytest.raises(TypeError, match="must be numeric"):
            store._validate_embedding([0.1, "x", 0.3])

    # --- _infer_field_type ---

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "boolean"),
            (10, "integer"),
            (1.5, "real"),
            ("text", "text"),
            ([1, 2], "text"),
            (None, "text"),
        ],
    )
    def test_infer_field_type(self, value, expected):
        assert Db2DocumentStore._infer_field_type(value) == expected
