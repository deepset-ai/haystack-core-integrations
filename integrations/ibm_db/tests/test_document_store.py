# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IBM DB2 Document Store using Haystack mixin tests."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
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
from haystack.utils import Secret

from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore
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
    Test IBMDb2DocumentStore using Haystack's standard mixin tests.

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

    def test_write_documents(self, document_store: IBMDb2DocumentStore):
        """Test write_documents() default behaviour required by the mixin."""
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1

    def test_write_documents_default_policy_none(self, document_store: IBMDb2DocumentStore):
        """Test that the default NONE policy rejects duplicate documents."""
        doc = Document(content="test doc")
        document_store.write_documents([doc])

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc])

    def test_connection_reuse(self, document_store: IBMDb2DocumentStore):
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

    def test_document_without_embedding(self, document_store: IBMDb2DocumentStore):
        """Test storing document without embedding."""
        doc = Document(id="no_emb", content="Document without embedding", meta={"test": True})
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_emb"})
        assert len(retrieved) == 1
        assert retrieved[0].embedding is None

    def test_document_without_content(self, document_store: IBMDb2DocumentStore):
        """Test storing document without content."""
        doc = Document(id="no_content", content=None, meta={"test": True}, embedding=[0.1] * 768)
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_content"})
        assert len(retrieved) == 1
        assert retrieved[0].content is None

    def test_close_and_reopen(self, document_store: IBMDb2DocumentStore):
        document_store.write_documents([Document(id="1", content="test", embedding=[0.1] * 768)])
        assert document_store.count_documents() == 1

        document_store.close()
        assert document_store._connection is None

        assert document_store.count_documents() == 1

    def test_complex_metadata(self, document_store: IBMDb2DocumentStore):
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

    def test_embedding_retrieval(self, document_store: IBMDb2DocumentStore):
        document_store.write_documents(
            [
                Document(id="a", content="alpha", embedding=[1.0, 0.0] + [0.0] * 766),
                Document(id="b", content="beta", embedding=[0.0, 1.0] + [0.0] * 766),
                Document(id="c", content="gamma", embedding=[0.9, 0.1] + [0.0] * 766),
            ]
        )

        results = document_store._embedding_retrieval([1.0, 0.0] + [0.0] * 766, top_k=2)

        assert [doc.id for doc in results] == ["a", "c"]
        assert all(doc.score is not None for doc in results)


class TestIBMDb2DocumentStoreUtilMethods:
    """Unit tests for pure utility methods: _parse_embedding, _validate_embedding, _infer_field_type."""

    # _parse_embedding
    def test_parse_embedding_none_returns_none(self):
        assert _parse_embedding(None) is None

    def test_parse_embedding_list_of_ints_converted_to_floats(self):
        assert _parse_embedding([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_parse_embedding_json_string_list(self):
        assert _parse_embedding("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]

    def test_parse_embedding_tuple_converted(self):
        assert _parse_embedding((0.5, 1.5)) == [0.5, 1.5]

    def test_parse_embedding_non_numeric_string_returns_none(self):
        assert _parse_embedding("not-a-vector") is None

    def test_parse_embedding_json_non_list_returns_none(self):
        assert _parse_embedding('{"a": 1}') is None

    def test_parse_embedding_non_iterable_returns_none(self):
        assert _parse_embedding(object()) is None

    # _validate_embedding
    def test_validate_embedding_none_allowed(self):
        IBMDb2DocumentStore._validate_embedding(None, allow_none=True)

    def test_validate_embedding_none_not_allowed_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            IBMDb2DocumentStore._validate_embedding(None, allow_none=False)

    def test_validate_embedding_non_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            IBMDb2DocumentStore._validate_embedding("not a list")

    def test_validate_embedding_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            IBMDb2DocumentStore._validate_embedding([])

    def test_validate_embedding_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError, match="must be numeric"):
            IBMDb2DocumentStore._validate_embedding([0.1, "x", 0.3])

    # _infer_field_type
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
        assert IBMDb2DocumentStore._infer_field_type(value) == expected


@pytest.fixture
def unit_store(monkeypatch) -> IBMDb2DocumentStore:
    monkeypatch.setenv("DB2_USERNAME", "db2inst1")
    monkeypatch.setenv("DB2_PASSWORD", "Passw0rd123!")
    return IBMDb2DocumentStore(
        database="testdb",
        hostname="localhost",
        username=Secret.from_env_var("DB2_USERNAME"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        embedding_dim=768,
        distance_metric="COSINE",
    )


@pytest.fixture
def mocked_store(unit_store) -> tuple:
    conn = MagicMock()
    cursor_cm = conn.cursor.return_value
    cursor_cm.__exit__.return_value = False
    cur = cursor_cm.__enter__.return_value
    unit_store._connection = conn
    unit_store._table_initialized = True
    return unit_store, conn, cur


_FILTER = {"operator": "==", "field": "meta.k", "value": "v"}


class TestIBMDb2DocumentStoreUnit:
    """Unit tests for IBMDb2DocumentStore that don't require a database."""

    def test_init_is_lazy_and_does_not_connect(self, unit_store):
        assert unit_store._connection is None
        assert unit_store._table_initialized is False

    def test_get_connection_applies_ssl_options_and_schema(self, monkeypatch):
        fake_conn = MagicMock()
        fake_conn.cursor.return_value.__exit__.return_value = False
        pconnect = Mock(return_value=fake_conn)
        monkeypatch.setattr(
            "haystack_integrations.document_stores.ibm_db.document_store.ibm_db_dbi.pconnect",
            pconnect,
        )
        store = IBMDb2DocumentStore(
            database="db",
            hostname="h",
            username=Secret.from_token("u"),
            password=Secret.from_token("p"),
            use_ssl=True,
            ssl_certificate="certs/db2.arm",
            connection_options={"foo": "bar"},
            schema="myschema",
        )
        store._table_initialized = True

        conn = store._get_connection()

        assert conn is fake_conn
        dsn = pconnect.call_args.kwargs["dsn"]
        assert "SECURITY=SSL" in dsn
        assert "SSLServerCertificate=certs/db2.arm" in dsn
        assert pconnect.call_args.kwargs["conn_options"]["foo"] == "bar"
        schema_cur = fake_conn.cursor.return_value.__enter__.return_value
        executed = [c.args[0] for c in schema_cur.execute.call_args_list]
        assert any("SET SCHEMA myschema" in s for s in executed)

    def test_ensure_table_exists_drops_and_creates_when_recreate(self, mocked_store):
        store, _, cur = mocked_store
        cur.execute.side_effect = [None, Exception("table not found"), None]

        store._ensure_table_exists(recreate=True)

        executed = [call.args[0] for call in cur.execute.call_args_list]
        assert any("DROP TABLE" in sql for sql in executed)
        assert any("CREATE TABLE" in sql for sql in executed)

    def test_ensure_table_exists_reraises_on_create_error(self, mocked_store):
        store, conn, cur = mocked_store
        cur.execute.side_effect = [Exception("table not found"), Exception("create failed")]

        with pytest.raises(Exception, match="create failed"):
            store._ensure_table_exists(recreate=False)

        conn.rollback.assert_called()

    def test_close(self, unit_store):
        mock_conn = Mock()
        unit_store._connection = mock_conn

        unit_store.close()

        mock_conn.close.assert_called_once()
        assert unit_store._connection is None

        unit_store.close()
        mock_conn.close.assert_called_once()

    def test_close_is_exception_safe(self, unit_store):
        mock_conn = Mock()
        mock_conn.close.side_effect = RuntimeError("boom")
        unit_store._connection = mock_conn

        unit_store.close()

        assert unit_store._connection is None

    def test_transaction_commits_on_success(self, mocked_store):
        store, conn, cur = mocked_store

        with store._transaction("unused on success") as transaction_cursor:
            assert transaction_cursor is cur

        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()

    def test_transaction_rolls_back_and_wraps_error(self, mocked_store):
        store, conn, _ = mocked_store
        db_error = "unique constraint violated"

        with (
            pytest.raises(DocumentStoreError, match=f"delete failed: {db_error}"),
            store._transaction("delete failed"),
        ):
            raise ValueError(db_error)

        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()

    def test_to_dict(self, unit_store):
        """Test serializing document store to dictionary without exposing credentials."""
        data = unit_store.to_dict()

        assert "type" in data
        assert data["type"] == "haystack_integrations.document_stores.ibm_db.document_store.IBMDb2DocumentStore"
        assert "init_parameters" in data

        init_params = data["init_parameters"]
        assert init_params["database"] == unit_store.database
        assert init_params["hostname"] == unit_store.hostname
        assert init_params["embedding_dim"] == 768
        assert init_params["distance_metric"] == "COSINE"
        assert init_params["recreate_table"] == unit_store.recreate_table
        # Credentials are serialized as env-var references, never as plaintext.
        assert init_params["password"] == {"type": "env_var", "env_vars": ["DB2_PASSWORD"], "strict": True}

    def test_from_dict(self, unit_store):
        """Test deserializing document store from dictionary."""
        data = unit_store.to_dict()

        new_store = IBMDb2DocumentStore.from_dict(data)

        assert new_store.embedding_dim == unit_store.embedding_dim
        assert new_store.distance_metric == unit_store.distance_metric
        assert new_store.database == unit_store.database
        assert new_store.hostname == unit_store.hostname
        assert isinstance(new_store.username, Secret)
        assert isinstance(new_store.password, Secret)

    def test_to_row_with_none_metadata(self, unit_store):
        """Test _to_row with None metadata."""
        doc = Document(id="1", content="test", meta=None, embedding=[0.1] * 768)
        row = unit_store._to_row(doc)

        assert row[0] == "1"  # id
        assert row[1] == "test"  # content
        assert row[2] == "{}"  # meta should be empty JSON object
        assert row[3] is not None  # embedding

    def test_to_row_with_none_embedding(self, unit_store):
        """Test _to_row with None embedding."""
        doc = Document(id="1", content="test", meta={"key": "value"}, embedding=None)
        row = unit_store._to_row(doc)

        assert row[0] == "1"  # id
        assert row[1] == "test"  # content
        assert '"key"' in row[2]  # meta JSON
        assert row[3] is None  # embedding should be None

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

    def test_build_where_clause_empty_filters(self, unit_store):
        """Test _build_where_clause with empty filters."""
        where_clause, params = unit_store._build_where_clause({})
        assert where_clause == ""
        assert params == []

    def test_write_documents_invalid_type(self, unit_store):
        """Test write_documents with invalid type."""
        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            unit_store.write_documents("not a list")

    def test_write_documents_invalid_document_type(self, unit_store):
        """Test write_documents with invalid document type in list."""
        with pytest.raises(ValueError, match="Expected Document objects"):
            unit_store.write_documents([{"id": "1", "content": "test"}])

    def test_write_documents_unsupported_policy(self, unit_store):
        """Test write_documents with unsupported duplicate policy."""
        unit_store._connection = Mock()
        unit_store._table_initialized = True
        doc = Document(id="1", content="test", embedding=[0.1] * 768)

        # Create a mock unsupported policy
        class UnsupportedPolicy:
            pass

        with pytest.raises(ValueError, match="Unsupported duplicate policy"):
            unit_store.write_documents([doc], policy=UnsupportedPolicy())

    def test_embedding_retrieval_returns_documents_with_scores(self, mocked_store):
        store, _, cur = mocked_store
        cur.fetchall.return_value = [
            ("d1", "content one", '{"k": "v"}', [0.1, 0.2, 0.3, 0.4], 0.05),
            ("d2", "content two", None, None, 0.2),
        ]

        docs = store._embedding_retrieval([0.1, 0.2, 0.3, 0.4], top_k=2)

        assert [d.id for d in docs] == ["d1", "d2"]
        assert docs[0].meta == {"k": "v"}
        assert docs[0].embedding == [0.1, 0.2, 0.3, 0.4]
        assert docs[0].score == 0.05
        assert docs[1].meta == {}
        assert docs[1].embedding is None

    def test_embedding_retrieval_appends_filter_and_null_check(self, mocked_store):
        store, _, cur = mocked_store
        cur.fetchall.return_value = []

        store._embedding_retrieval([0.1, 0.2], filters=_FILTER, top_k=5)

        executed_sql = cur.execute.call_args.args[0]
        assert "AND embedding IS NOT NULL" in executed_sql

    @pytest.mark.parametrize(
        "fetch_error",
        [
            Exception("SQL0801N division by zero"),
            Exception("Division by zero"),
        ],
    )
    def test_embedding_retrieval_returns_empty_on_zero_vector_error(self, mocked_store, fetch_error):
        store, _, cur = mocked_store
        cur.fetchall.side_effect = fetch_error

        assert store._embedding_retrieval([0.1, 0.2], top_k=1) == []

    def test_embedding_retrieval_reraises_unexpected_fetch_error(self, mocked_store):
        store, _, cur = mocked_store
        cur.fetchall.side_effect = RuntimeError("connection reset")

        with pytest.raises(RuntimeError, match="connection reset"):
            store._embedding_retrieval([0.1, 0.2], top_k=1)
