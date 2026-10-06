# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IBM DB2 Document Store using Haystack mixin tests."""

import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
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

    @pytest.mark.parametrize(
        "metric",
        ["COSINE", "DOT", "EUCLIDEAN", "EUCLIDEAN_SQUARED", "HAMMING", "MANHATTAN"],
    )
    def test_all_distance_metrics_return_ranked_results(self, metric: str, connection_config):
        """
        Verify that _embedding_retrieval works correctly for each of the 6 Db2
        distance metrics: correct ranking, finite scores, top_k respected.

        A fresh store with embedding_dim=4 is created for each metric so the
        test is independent of the default `document_store` fixture.
        All Db2 distance metrics return scores ordered ASC (smaller = more similar).
        """
        table_name = f"haystack_metric_{metric.lower()}_{sys.version_info.major}_{sys.version_info.minor}"
        store = IBMDb2DocumentStore(
            **connection_config,
            table_name=table_name,
            embedding_dim=4,
            distance_metric=metric,
            recreate_table=True,
        )

        try:
            # Embeddings chosen so ranking is unambiguous for every metric:
            #   most_similar  == query  → distance 0 (or near 0) for all metrics
            #   second_best   is close  → small distance
            #   least_similar is far    → large distance
            query = [1.0, 1.0, 1.0, 1.0]
            most_similar = [1.0, 1.0, 1.0, 1.0]
            second_best = [1.0, 1.0, 0.5, 0.5]
            least_similar = [0.0, 0.0, 0.0, 1.0]

            docs = [
                Document(content="most_similar", embedding=most_similar),
                Document(content="second_best", embedding=second_best),
                Document(content="least_similar", embedding=least_similar),
            ]
            store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

            results = store._embedding_retrieval(query_embedding=query, top_k=2)

            # Exactly top_k documents returned
            assert len(results) == 2, f"[{metric}] Expected 2 results, got {len(results)}"

            # most_similar must be in the top-2; least_similar must be excluded
            contents = [r.content for r in results]
            assert "most_similar" in contents, f"[{metric}] 'most_similar' missing from top-2: {contents}"
            assert "least_similar" not in contents, f"[{metric}] 'least_similar' appeared in top-2: {contents}"

            # Scores must be finite floats and ordered ASC (ORDER BY score ASC)
            for doc in results:
                assert doc.score is not None, f"[{metric}] score is None"
                assert math.isfinite(doc.score), f"[{metric}] non-finite score: {doc.score}"
            assert results[0].content == "most_similar", (
                f"[{metric}] Expected 'most_similar' at rank-1, got '{results[0].content}' "
                f"(scores: {[r.score for r in results]})"
            )
            assert results[0].score <= results[1].score, (
                f"[{metric}] Score ordering wrong: {results[0].score} > {results[1].score}"
            )

        finally:
            try:
                conn = store._get_connection()
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE {store.table_name}")
                    conn.commit()
            except Exception:
                pass
            store.close()

    @pytest.mark.parametrize(
        "metric",
        ["COSINE", "DOT", "EUCLIDEAN", "EUCLIDEAN_SQUARED", "HAMMING", "MANHATTAN"],
    )
    def test_all_distance_metrics_respect_filters(self, metric: str, connection_config):
        """
        Verify that metadata filters work correctly alongside each distance metric.
        """
        table_name = f"haystack_metric_flt_{metric.lower()}_{sys.version_info.major}_{sys.version_info.minor}"
        store = IBMDb2DocumentStore(
            **connection_config,
            table_name=table_name,
            embedding_dim=4,
            distance_metric=metric,
            recreate_table=True,
        )

        try:
            query = [1.0, 1.0, 1.0, 1.0]
            docs = [
                Document(content="in_1", embedding=[1.0, 1.0, 1.0, 1.0], meta={"cat": "A"}),
                Document(content="in_2", embedding=[0.9, 0.9, 0.9, 0.9], meta={"cat": "A"}),
                Document(content="out", embedding=[1.0, 1.0, 1.0, 1.0], meta={"cat": "B"}),
            ]
            store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

            filters = {"field": "meta.cat", "operator": "==", "value": "A"}
            results = store._embedding_retrieval(query_embedding=query, top_k=5, filters=filters)

            assert len(results) == 2, f"[{metric}] Expected 2 filtered results, got {len(results)}"
            for doc in results:
                assert doc.meta.get("cat") == "A", f"[{metric}] Doc with wrong category returned: {doc.meta}"

        finally:
            try:
                conn = store._get_connection()
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE {store.table_name}")
                    conn.commit()
            except Exception:
                pass
            store.close()

    def test_get_metadata_field_unique_values_pagination(self, document_store: IBMDb2DocumentStore):
        docs = [
            Document(content=f"doc {i}", meta={"category": c}) for i, c in enumerate(["A", "B", "A", "C", "B", "A"])
        ]
        document_store.write_documents(docs)

        page1, total1 = document_store.get_metadata_field_unique_values("category", from_=0, size=2)
        page2, total2 = document_store.get_metadata_field_unique_values("category", from_=2, size=2)

        assert len(page1) == 2
        assert len(page2) == 1
        assert total1 == total2 == 3
        assert not set(page1).intersection(page2)
        assert set(page1) | set(page2) == {"A", "B", "C"}

        all_values, total = document_store.get_metadata_field_unique_values("category", from_=0, size=100)
        assert set(all_values) == {"A", "B", "C"}
        assert total == 3

        beyond, total_beyond = document_store.get_metadata_field_unique_values("category", from_=10, size=10)
        assert beyond == []
        assert total_beyond == 3

    def test_get_metadata_field_unique_values_invalid_field_name(self, document_store: IBMDb2DocumentStore):
        with pytest.raises(ValueError, match="Invalid metadata field name"):
            document_store.get_metadata_field_unique_values("field' OR '1'='1")

    def test_get_metadata_field_unique_values_hyphenated_field_name(self, document_store: IBMDb2DocumentStore):
        docs = [
            Document(content="one", meta={"some-field": "A"}),
            Document(content="two", meta={"some-field": "B"}),
            Document(content="three", meta={"some-field": "A"}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values("some-field")
        assert set(values) == {"A", "B"}
        assert total_count == 2

    def test_case_insensitive_and_special_characters_are_allowed(self, document_store: IBMDb2DocumentStore):
        docs = [
            Document(meta={"category": "100% complete"}),
            Document(meta={"category": "plain"}),
            Document(meta={"category": "under_score"}),
            Document(meta={"category": "underXscore"}),
        ]
        document_store.write_documents(docs)

        values, total = document_store.get_metadata_field_unique_values("category", search_term="%")
        assert values == ["100% complete"]
        assert total == 1

        values, total = document_store.get_metadata_field_unique_values("category", search_term="_")
        assert values == ["under_score"]
        assert total == 1


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

    # _normalize_metadata_field_name
    def test_normalize_metadata_field_name_strips_meta_prefix(self):
        assert IBMDb2DocumentStore._normalize_metadata_field_name("meta.category") == "category"

    def test_normalize_metadata_field_name_without_prefix_unchanged(self):
        assert IBMDb2DocumentStore._normalize_metadata_field_name("category") == "category"

    def test_normalize_metadata_field_name_allows_underscore_and_hyphen(self):
        assert IBMDb2DocumentStore._normalize_metadata_field_name("meta.some_field-name") == "some_field-name"

    @pytest.mark.parametrize(
        "bad_field",
        [
            "field@invalid",
            "field with spaces",
            "field;drop",
            "meta.field!",
            "field' OR '1'='1",
            "../etc/passwd",
            "field/* comment */",
        ],
    )
    def test_normalize_metadata_field_name_rejects_invalid_chars(self, bad_field):
        with pytest.raises(ValueError, match="Invalid metadata field name"):
            IBMDb2DocumentStore._normalize_metadata_field_name(bad_field)


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

    @pytest.mark.parametrize(
        "metric",
        [
            "COSINE",
            "DOT",
            "EUCLIDEAN",
            "EUCLIDEAN_SQUARED",
            "HAMMING",
            "MANHATTAN",
        ],
    )
    def test_supported_distance_metrics(self, metric, monkeypatch):
        """Test that all 6 supported Db2 distance metrics can be configured and serialized."""
        monkeypatch.setenv("DB2_USERNAME", "db2inst1")
        monkeypatch.setenv("DB2_PASSWORD", "Passw0rd123!")
        store = IBMDb2DocumentStore(
            database="testdb",
            hostname="localhost",
            username=Secret.from_env_var("DB2_USERNAME"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            distance_metric=metric,
        )
        assert store.distance_metric == metric
        data = store.to_dict()
        assert data["init_parameters"]["distance_metric"] == metric
        restored = IBMDb2DocumentStore.from_dict(data)
        assert restored.distance_metric == metric

    def test_invalid_distance_metric_raises(self, monkeypatch):
        """Test that an invalid distance_metric raises ValueError at construction time."""
        monkeypatch.setenv("DB2_USERNAME", "db2inst1")
        monkeypatch.setenv("DB2_PASSWORD", "Passw0rd123!")
        with pytest.raises(ValueError, match="Invalid distance_metric"):
            IBMDb2DocumentStore(
                database="testdb",
                hostname="localhost",
                username=Secret.from_env_var("DB2_USERNAME"),
                password=Secret.from_env_var("DB2_PASSWORD"),
                distance_metric="L2",  # type: ignore[arg-type]
            )

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
        "metric",
        [
            "COSINE",
            "DOT",
            "EUCLIDEAN",
            "EUCLIDEAN_SQUARED",
            "HAMMING",
            "MANHATTAN",
        ],
    )
    def test_embedding_retrieval_uses_configured_metric(self, mocked_store, metric):
        store, _, cur = mocked_store
        store.distance_metric = metric
        cur.fetchall.return_value = []

        store._embedding_retrieval([0.1, 0.2], top_k=5)

        executed_sql = cur.execute.call_args.args[0]
        expected_fragment = (
            f"VECTOR_DISTANCE(embedding, VECTOR(CAST(? AS CLOB(100000)), {store.embedding_dim}, FLOAT32), {metric})"
        )
        assert expected_fragment in executed_sql

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

    def test_get_metadata_field_unique_values_builds_paginated_sql(self, mocked_store):
        store, _, cur = mocked_store
        cur.fetchone.return_value = (2,)
        # JSON_QUERY returns the literal JSON form of a string - quoted - unlike JSON_VALUE which
        # dequotes it; json.loads() below relies on that to tell a string apart from a number.
        cur.fetchall.return_value = [('"A"',), ('"B"',)]

        values, total_count = store.get_metadata_field_unique_values("meta.category", from_=5, size=10)

        assert values == ["A", "B"]
        assert total_count == 2

        count_sql, count_params = cur.execute.call_args_list[0].args
        select_sql, select_params = cur.execute.call_args_list[1].args

        assert "COUNT(DISTINCT value)" in count_sql
        assert "$.category" in count_sql
        assert count_params == []

        assert "SELECT DISTINCT value" in select_sql
        assert "ORDER BY value OFFSET ? ROWS FETCH FIRST ? ROWS ONLY" in select_sql
        assert select_params == [5, 10]

    def test_get_metadata_field_unique_values_applies_search_term(self, mocked_store):
        store, _, cur = mocked_store
        cur.fetchone.return_value = (1,)
        cur.fetchall.return_value = [('"Python-based"',)]

        store.get_metadata_field_unique_values("meta.category", search_term="python")

        count_sql, count_params = cur.execute.call_args_list[0].args
        select_sql, select_params = cur.execute.call_args_list[1].args

        assert "LOCATE(UPPER(?), UPPER(value)) > 0" in count_sql
        assert count_params == ["python"]
        assert "LOCATE(UPPER(?), UPPER(value)) > 0" in select_sql
        assert select_params == ["python", 0, 10]

    def test_get_metadata_field_unique_values_rejects_invalid_field_name(self, mocked_store):
        store, _, _ = mocked_store

        with pytest.raises(ValueError, match="Invalid metadata field name"):
            store.get_metadata_field_unique_values("field' OR '1'='1")

    def test_get_metadata_field_unique_values_handles_no_rows(self, mocked_store):
        store, _, cur = mocked_store
        cur.fetchone.return_value = None
        cur.fetchall.return_value = []

        values, total_count = store.get_metadata_field_unique_values("meta.category")

        assert values == []
        assert total_count == 0


class TestIBMDb2DocumentStoreVectorIndex:
    """Unit tests for the create_vector_index / _ensure_vector_index_exists behaviour."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_store(
        self, monkeypatch, *, create_vector_index: bool = True, distance_metric: str = "COSINE"
    ) -> IBMDb2DocumentStore:
        monkeypatch.setenv("DB2_USERNAME", "db2inst1")
        monkeypatch.setenv("DB2_PASSWORD", "Passw0rd123!")
        store = IBMDb2DocumentStore(
            database="testdb",
            hostname="localhost",
            username=Secret.from_env_var("DB2_USERNAME"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            embedding_dim=4,
            distance_metric=distance_metric,
            create_vector_index=create_vector_index,
        )
        conn = MagicMock()
        cursor_cm = conn.cursor.return_value
        cursor_cm.__exit__.return_value = False
        store._connection = conn
        store._table_initialized = True
        return store, conn, cursor_cm.__enter__.return_value

    # ------------------------------------------------------------------
    # __init__ and to_dict / from_dict
    # ------------------------------------------------------------------

    def test_create_vector_index_defaults_to_false(self, monkeypatch):
        monkeypatch.setenv("DB2_USERNAME", "u")
        monkeypatch.setenv("DB2_PASSWORD", "p")
        store = IBMDb2DocumentStore(database="d", hostname="h")
        assert store.create_vector_index is False

    def test_create_vector_index_roundtrips_to_dict(self, monkeypatch):
        monkeypatch.setenv("DB2_USERNAME", "u")
        monkeypatch.setenv("DB2_PASSWORD", "p")
        store = IBMDb2DocumentStore(database="d", hostname="h", create_vector_index=True)
        data = store.to_dict()
        assert data["init_parameters"]["create_vector_index"] is True
        restored = IBMDb2DocumentStore.from_dict(data)
        assert restored.create_vector_index is True

    # ------------------------------------------------------------------
    # Unsupported distance metric
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("metric", ["MANHATTAN", "DOT", "HAMMING"])
    def test_unsupported_metric_logs_warning_and_skips(self, monkeypatch, caplog, metric):
        store, conn, _ = self._make_store(monkeypatch, distance_metric=metric)
        with caplog.at_level(logging.WARNING, logger="haystack_integrations.document_stores.ibm_db.document_store"):
            store._ensure_vector_index_exists(conn)

        assert "not supported" in caplog.text
        # No DDL should have been executed
        conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()

    # ------------------------------------------------------------------
    # Index already exists — idempotency
    # ------------------------------------------------------------------

    def test_skips_creation_when_index_already_exists(self, monkeypatch):
        store, conn, cur = self._make_store(monkeypatch)
        # Simulate CURRENT SCHEMA query then index-exists query returning count=1
        cur.fetchone.side_effect = [("DB2INST1",), (1,)]

        store._ensure_vector_index_exists(conn)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        assert not any("CREATE VECTOR INDEX" in sql for sql in executed_sqls)
        conn.commit.assert_not_called()

    # ------------------------------------------------------------------
    # Index does not exist — creation path
    # ------------------------------------------------------------------

    def test_creates_index_when_absent(self, monkeypatch):
        store, conn, cur = self._make_store(monkeypatch)
        # CURRENT SCHEMA → count=0 (no existing index)
        cur.fetchone.side_effect = [("DB2INST1",), (0,)]

        store._ensure_vector_index_exists(conn)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        create_sqls = [s for s in executed_sqls if "CREATE VECTOR INDEX" in s]
        assert len(create_sqls) == 1
        assert "haystack_documents_embedding_idx" in create_sqls[0]
        assert "WITH DISTANCE COSINE" in create_sqls[0]
        conn.commit.assert_called_once()

    def test_index_name_uses_table_name(self, monkeypatch):
        monkeypatch.setenv("DB2_USERNAME", "u")
        monkeypatch.setenv("DB2_PASSWORD", "p")
        store = IBMDb2DocumentStore(
            database="d", hostname="h",
            table_name="my_custom_table",
            create_vector_index=True,
        )
        conn = MagicMock()
        cursor_cm = conn.cursor.return_value
        cursor_cm.__exit__.return_value = False
        cur = cursor_cm.__enter__.return_value
        cur.fetchone.side_effect = [("MYSCHEMA",), (0,)]

        store._ensure_vector_index_exists(conn)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        create_sqls = [s for s in executed_sqls if "CREATE VECTOR INDEX" in s]
        assert len(create_sqls) == 1
        assert "my_custom_table_embedding_idx" in create_sqls[0]
        assert "ON my_custom_table" in create_sqls[0]

    @pytest.mark.parametrize("metric", ["EUCLIDEAN", "EUCLIDEAN_SQUARED"])
    def test_supported_metric_used_in_index_ddl(self, monkeypatch, metric):
        store, conn, cur = self._make_store(monkeypatch, distance_metric=metric)
        cur.fetchone.side_effect = [("DB2INST1",), (0,)]

        store._ensure_vector_index_exists(conn)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        create_sqls = [s for s in executed_sqls if "CREATE VECTOR INDEX" in s]
        assert f"WITH DISTANCE {metric}" in create_sqls[0]

    # ------------------------------------------------------------------
    # Schema-qualified catalog lookup
    # ------------------------------------------------------------------

    def test_catalog_query_includes_schema_when_set(self, monkeypatch):
        monkeypatch.setenv("DB2_USERNAME", "u")
        monkeypatch.setenv("DB2_PASSWORD", "p")
        store = IBMDb2DocumentStore(
            database="d", hostname="h",
            schema="MYSCHEMA",
            create_vector_index=True,
        )
        conn = MagicMock()
        cursor_cm = conn.cursor.return_value
        cursor_cm.__exit__.return_value = False
        cur = cursor_cm.__enter__.return_value
        # schema is set on store, so no CURRENT SCHEMA query; count=0 → create
        cur.fetchone.return_value = (0,)

        store._ensure_vector_index_exists(conn)

        # The SYSCAT query should include TABSCHEMA parameter
        syscat_call = cur.execute.call_args_list[0]
        syscat_sql = syscat_call.args[0]
        syscat_params = syscat_call.args[1]
        assert "TABSCHEMA" in syscat_sql
        assert "MYSCHEMA" in syscat_params

    # ------------------------------------------------------------------
    # Driver error wrapping
    # ------------------------------------------------------------------

    def test_driver_error_raises_document_store_error(self, monkeypatch):
        store, conn, cur = self._make_store(monkeypatch)
        cur.fetchone.side_effect = [("DB2INST1",), (0,)]
        cur.execute.side_effect = [None, None, Exception("SQL1585N unsupported platform")]

        with pytest.raises(DocumentStoreError, match="Failed to create vector index"):
            store._ensure_vector_index_exists(conn)

        conn.rollback.assert_called_once()

    # ------------------------------------------------------------------
    # _ensure_table_exists integration
    # ------------------------------------------------------------------

    def test_create_table_uses_not_null_embedding_when_index_enabled(self, monkeypatch):
        """embedding column must be NOT NULL when create_vector_index=True."""
        store, _, cur = self._make_store(monkeypatch)
        # Table doesn't exist — SELECT COUNT(*) raises → CREATE TABLE is executed
        cur.execute.side_effect = [
            Exception("table not found"),  # SELECT COUNT(*) — table absent
            None,                           # CREATE TABLE
            None,                           # VALUES CURRENT SCHEMA
            None,                           # SELECT COUNT(*) from SYSCAT
            None,                           # CREATE VECTOR INDEX
        ]
        cur.fetchone.side_effect = [("DB2INST1",), (0,)]

        store._ensure_table_exists(recreate=False)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        create_table_sqls = [s for s in executed_sqls if "CREATE TABLE" in s]
        assert len(create_table_sqls) == 1
        assert "NOT NULL" in create_table_sqls[0], "embedding column must be NOT NULL"

    def test_create_table_allows_null_embedding_when_index_disabled(self, monkeypatch):
        """embedding column must remain nullable when create_vector_index=False."""
        store, _, cur = self._make_store(monkeypatch, create_vector_index=False)
        cur.execute.side_effect = [
            Exception("table not found"),  # SELECT COUNT(*) — table absent
            None,                           # CREATE TABLE
        ]

        store._ensure_table_exists(recreate=False)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        create_table_sqls = [s for s in executed_sqls if "CREATE TABLE" in s]
        assert len(create_table_sqls) == 1
        # Trailing whitespace from empty nullability string is acceptable; just no NOT NULL
        assert "NOT NULL" not in create_table_sqls[0].split("FLOAT32")[1], (
            "embedding column must be nullable when create_vector_index=False"
        )

    def test_ensure_table_exists_calls_vector_index_when_enabled(self, monkeypatch):
        store, _, cur = self._make_store(monkeypatch)
        # The table-existence check in _ensure_table_exists uses execute() + exception
        # (no fetchone call).  fetchone is called only inside _ensure_vector_index_exists:
        #   1. CURRENT SCHEMA query → ("DB2INST1",)
        #   2. SYSCAT.INDEXES count → (0,) meaning index absent → triggers CREATE
        cur.execute.return_value = None
        cur.fetchone.side_effect = [("DB2INST1",), (0,)]

        store._ensure_table_exists(recreate=False)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        assert any("CREATE VECTOR INDEX" in sql for sql in executed_sqls)

    def test_ensure_table_exists_skips_vector_index_when_disabled(self, monkeypatch):
        store, _, cur = self._make_store(monkeypatch, create_vector_index=False)
        cur.execute.return_value = None

        store._ensure_table_exists(recreate=False)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        assert not any("CREATE VECTOR INDEX" in sql for sql in executed_sqls)

    def test_ensure_table_exists_recreate_drops_then_creates_index(self, monkeypatch):
        """recreate=True drops + recreates the table and then creates the vector index."""
        store, _, cur = self._make_store(monkeypatch)
        # DROP TABLE succeeds, then SELECT COUNT(*) raises (table gone), then
        # CREATE TABLE succeeds.  After that, _ensure_vector_index_exists runs:
        #   CURRENT SCHEMA → index absent → CREATE VECTOR INDEX.
        cur.execute.side_effect = [
            None,                        # DROP TABLE
            Exception("table not found"),# SELECT COUNT(*) — table doesn't exist yet
            None,                        # CREATE TABLE
            None,                        # VALUES CURRENT SCHEMA
            None,                        # SELECT COUNT(*) from SYSCAT
            None,                        # CREATE VECTOR INDEX
        ]
        cur.fetchone.side_effect = [("DB2INST1",), (0,)]

        store._ensure_table_exists(recreate=True)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        assert any("DROP TABLE" in sql for sql in executed_sqls)
        assert any("CREATE TABLE" in sql for sql in executed_sqls)
        assert any("CREATE VECTOR INDEX" in sql for sql in executed_sqls)

    def test_index_name_strips_schema_prefix(self, monkeypatch):
        """Index name must use only the bare table name, not a schema-qualified one."""
        monkeypatch.setenv("DB2_USERNAME", "u")
        monkeypatch.setenv("DB2_PASSWORD", "p")
        store = IBMDb2DocumentStore(
            database="d", hostname="h",
            table_name="myschema.mytable",
            create_vector_index=True,
        )
        conn = MagicMock()
        cursor_cm = conn.cursor.return_value
        cursor_cm.__exit__.return_value = False
        cur = cursor_cm.__enter__.return_value
        cur.fetchone.side_effect = [("MYSCHEMA",), (0,)]

        store._ensure_vector_index_exists(conn)

        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        create_sqls = [s for s in executed_sqls if "CREATE VECTOR INDEX" in s]
        assert len(create_sqls) == 1
        # Index name must be "mytable_embedding_idx", not "myschema.mytable_embedding_idx"
        assert "mytable_embedding_idx" in create_sqls[0]
        assert "myschema.mytable_embedding_idx" not in create_sqls[0]


@pytest.mark.integration
class TestVectorIndexIntegration:
    """
    Integration tests for create_vector_index against a real Db2 server.

    These tests require the Db2 server to be version 12.1.5 or later on LinuxAMD64.
    They use a dedicated table name to avoid interfering with the main test suite.
    """

    _TABLE = "haystack_vidx_integ_test"

    @pytest.fixture(autouse=True)
    def _cleanup(self, connection_config):
        """Drop the test table before and after each test."""
        def _drop():
            try:
                store = IBMDb2DocumentStore(**connection_config, table_name=self._TABLE)
                conn = store._get_connection()
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE {self._TABLE}")
                    conn.commit()
                store.close()
            except Exception:
                pass

        _drop()
        yield
        _drop()

    def test_create_vector_index_creates_index_in_syscat(self, connection_config):
        """create_vector_index=True should result in an entry in SYSCAT.INDEXES."""
        store = IBMDb2DocumentStore(
            **connection_config,
            table_name=self._TABLE,
            embedding_dim=4,
            distance_metric="COSINE",
            recreate_table=True,
            create_vector_index=True,
        )
        store._get_connection()  # triggers table + index creation

        conn = store._connection
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM SYSCAT.INDEXES "
                "WHERE UPPER(INDNAME) = UPPER(?) AND UPPER(TABNAME) = UPPER(?)",
                (f"{self._TABLE}_embedding_idx", self._TABLE),
            )
            row = cur.fetchone()

        store.close()
        assert row is not None and row[0] == 1, "Vector index not found in SYSCAT.INDEXES"

    def test_create_vector_index_is_idempotent(self, connection_config):
        """Initialising the same store twice must not raise and must leave exactly one index."""
        kwargs = {
            **connection_config,
            "table_name": self._TABLE,
            "embedding_dim": 4,
            "distance_metric": "COSINE",
            "create_vector_index": True,
        }

        store1 = IBMDb2DocumentStore(**kwargs, recreate_table=True)
        store1._get_connection()
        store1.close()

        # Second init — table already exists, index already exists; must be silent.
        store2 = IBMDb2DocumentStore(**kwargs, recreate_table=False)
        store2._get_connection()

        conn = store2._connection
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM SYSCAT.INDEXES "
                "WHERE UPPER(INDNAME) = UPPER(?) AND UPPER(TABNAME) = UPPER(?)",
                (f"{self._TABLE}_embedding_idx", self._TABLE),
            )
            row = cur.fetchone()

        store2.close()
        assert row is not None and row[0] == 1, "Expected exactly one index after two inits"

    def test_create_vector_index_rebuilt_after_recreate_table(self, connection_config):
        """recreate_table=True drops the table (and its index) then rebuilds both."""
        kwargs = {
            **connection_config,
            "table_name": self._TABLE,
            "embedding_dim": 4,
            "distance_metric": "COSINE",
            "create_vector_index": True,
        }

        # First init — creates table + index.
        store1 = IBMDb2DocumentStore(**kwargs, recreate_table=True)
        store1._get_connection()
        store1.close()

        # Second init with recreate_table=True — must drop + recreate table and index.
        store2 = IBMDb2DocumentStore(**kwargs, recreate_table=True)
        store2._get_connection()

        conn = store2._connection
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM SYSCAT.INDEXES "
                "WHERE UPPER(INDNAME) = UPPER(?) AND UPPER(TABNAME) = UPPER(?)",
                (f"{self._TABLE}_embedding_idx", self._TABLE),
            )
            row = cur.fetchone()

        store2.close()
        assert row is not None and row[0] == 1, "Vector index not found after recreate_table=True"
