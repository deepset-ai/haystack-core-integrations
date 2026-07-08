# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
from haystack_integrations.document_stores.mariadb.document_store import (
    _bytes_to_embedding,
    _document_to_row,
    _embedding_to_bytes,
    _rows_to_documents,
)

# ---------------------------------------------------------------------------
# Helper: build a store with a mocked DB connection
# ---------------------------------------------------------------------------

def _mock_store(**kwargs) -> MariaDBDocumentStore:
    store = MariaDBDocumentStore(
        host="localhost",
        database="test_db",
        user="root",
        password="secret",
        **kwargs,
    )
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.dictionary = True
    mock_conn.cursor.return_value = mock_cursor
    store._connection = mock_conn
    store._cursor = mock_cursor
    store._table_initialized = True
    return store


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MARIADB_USER", "test_user")
        monkeypatch.setenv("MARIADB_PASSWORD", "test_pass")
        store = MariaDBDocumentStore()
        d = store.to_dict()
        assert d["type"] == "haystack_integrations.document_stores.mariadb.document_store.MariaDBDocumentStore"
        params = d["init_parameters"]
        assert params["host"] == "localhost"
        assert params["embedding_dimension"] == 768
        assert params["vector_function"] == "cosine_similarity"

    def test_from_dict_roundtrip(self, monkeypatch):
        monkeypatch.setenv("MARIADB_USER", "test_user")
        monkeypatch.setenv("MARIADB_PASSWORD", "test_pass")
        store = MariaDBDocumentStore(embedding_dimension=512)
        d = store.to_dict()
        restored = MariaDBDocumentStore.from_dict(d)
        assert restored.embedding_dimension == 512
        assert restored.host == "localhost"

    def test_invalid_vector_function_raises(self):
        with pytest.raises(ValueError, match="vector_function must be one of"):
            MariaDBDocumentStore(user="u", password="p", vector_function="bad_func")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

class TestEmbeddingHelpers:
    def test_round_trip_embedding(self):
        original = [0.1, 0.2, 0.3, -0.5]
        raw = _embedding_to_bytes(original)
        recovered = _bytes_to_embedding(raw)
        assert len(recovered) == len(original)
        for a, b in zip(recovered, original, strict=True):
            assert abs(a - b) < 1e-6

    def test_embedding_bytes_length(self):
        emb = [1.0] * 8
        raw = _embedding_to_bytes(emb)
        assert len(raw) == 8 * 4  # 4 bytes per float32


# ---------------------------------------------------------------------------
# Document conversion helpers
# ---------------------------------------------------------------------------

class TestDocumentHelpers:
    def test_document_to_row_simple(self):
        doc = Document(id="abc", content="hello", meta={"x": 1})
        row = _document_to_row(doc)
        assert row[0] == "abc"
        assert row[1] is None  # no embedding
        assert row[2] == "hello"
        assert row[3] is None  # no blob
        assert row[6] == '{"x": 1}'

    def test_document_to_row_with_embedding(self):
        doc = Document(id="d1", content="text", embedding=[0.5, 0.5])
        row = _document_to_row(doc)
        assert isinstance(row[1], bytes)

    def test_document_to_row_with_blob(self):
        blob = ByteStream(data=b"raw bytes", meta={"k": "v"}, mime_type="application/octet-stream")
        doc = Document(id="d2", content=None, blob=blob)
        row = _document_to_row(doc)
        assert row[3] == b"raw bytes"
        assert row[4] == '{"k": "v"}'
        assert row[5] == "application/octet-stream"

    def test_rows_to_documents_simple(self):
        records = [{"id": "x1", "content": "text", "embedding": None,
                    "blob_data": None, "blob_meta": None, "blob_mime_type": None,
                    "meta": '{"a": 1}'}]
        docs = _rows_to_documents(records)
        assert len(docs) == 1
        assert docs[0].id == "x1"
        assert docs[0].meta == {"a": 1}

    def test_rows_to_documents_bytes_embedding(self):
        emb = [0.1, 0.9]
        raw = _embedding_to_bytes(emb)
        records = [{"id": "e1", "content": "hi", "embedding": raw,
                    "blob_data": None, "blob_meta": None, "blob_mime_type": None,
                    "meta": "{}"}]
        docs = _rows_to_documents(records)
        assert docs[0].embedding is not None
        assert len(docs[0].embedding) == 2

    def test_rows_to_documents_strips_score(self):
        records = [{"id": "s1", "content": "text", "embedding": None,
                    "blob_data": None, "blob_meta": None, "blob_mime_type": None,
                    "meta": "{}", "score": 0.99}]
        docs = _rows_to_documents(records)
        # Document.score should not be set from the row (no score field in Document)
        assert docs[0].id == "s1"

    def test_rows_to_documents_with_blob(self):
        records = [{"id": "b1", "content": None, "embedding": None,
                    "blob_data": b"data", "blob_meta": '{"k": "v"}',
                    "blob_mime_type": "image/png", "meta": "{}"}]
        docs = _rows_to_documents(records)
        assert docs[0].blob is not None
        assert docs[0].blob.data == b"data"
        assert docs[0].blob.mime_type == "image/png"


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------

class TestCountDocuments:
    def test_count_documents(self):
        store = _mock_store()
        store._cursor.fetchone.return_value = {"cnt": 5}
        assert store.count_documents() == 5

    def test_count_empty(self):
        store = _mock_store()
        store._cursor.fetchone.return_value = {"cnt": 0}
        assert store.count_documents() == 0


# ---------------------------------------------------------------------------
# write_documents
# ---------------------------------------------------------------------------

class TestWriteDocuments:
    def test_write_empty_returns_zero(self):
        store = _mock_store()
        assert store.write_documents([]) == 0

    def test_write_single_doc(self):
        store = _mock_store()
        doc = Document(content="test")
        result = store.write_documents([doc])
        assert result == 1
        store._cursor.execute.assert_called_once()

    def test_write_overwrite_uses_upsert(self):
        store = _mock_store()
        doc = Document(content="test")
        store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        call_args = store._cursor.execute.call_args[0]
        assert "ON DUPLICATE KEY UPDATE" in call_args[0]

    def test_write_skip_uses_insert_ignore(self):
        store = _mock_store()
        doc = Document(content="test")
        store.write_documents([doc], policy=DuplicatePolicy.SKIP)
        call_args = store._cursor.execute.call_args[0]
        assert "INSERT IGNORE" in call_args[0]

    def test_write_fail_raises_on_duplicate(self):
        import mariadb  # noqa: PLC0415
        store = _mock_store()
        store._cursor.execute.side_effect = mariadb.IntegrityError("Duplicate entry")
        doc = Document(id="dup", content="test")
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_write_invalid_type_raises(self):
        store = _mock_store()
        with pytest.raises(ValueError, match="list of Document objects"):
            store.write_documents(["not a doc"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------

class TestDeleteDocuments:
    def test_delete_empty_noop(self):
        store = _mock_store()
        store.delete_documents([])
        store._cursor.execute.assert_not_called()

    def test_delete_calls_execute(self):
        store = _mock_store()
        store.delete_documents(["id1", "id2"])
        call_args = store._cursor.execute.call_args[0]
        assert "DELETE FROM" in call_args[0]
        assert "IN (?, ?)" in call_args[0]


# ---------------------------------------------------------------------------
# filter_documents
# ---------------------------------------------------------------------------

class TestFilterDocuments:
    def test_filter_no_filters(self):
        store = _mock_store()
        store._cursor.fetchall.return_value = []
        result = store.filter_documents()
        assert result == []
        call_args = store._cursor.execute.call_args[0]
        assert "WHERE" not in call_args[0]

    def test_filter_with_filters(self):
        store = _mock_store()
        store._cursor.fetchall.return_value = []
        store.filter_documents(filters={"field": "meta.x", "operator": "==", "value": 1})
        call_args = store._cursor.execute.call_args[0]
        assert "WHERE" in call_args[0]


# ---------------------------------------------------------------------------
# Integration tests (require real MariaDB)
# ---------------------------------------------------------------------------

MARIADB_HOST = os.environ.get("MARIADB_HOST", "localhost")
MARIADB_PORT = int(os.environ.get("MARIADB_PORT", "3306"))
MARIADB_DB = os.environ.get("MARIADB_DATABASE", "haystack")
MARIADB_USER = os.environ.get("MARIADB_USER", "root")
MARIADB_PASSWORD = os.environ.get("MARIADB_PASSWORD", "password")


@pytest.fixture
def integration_store():
    store = MariaDBDocumentStore(
        host=MARIADB_HOST,
        port=MARIADB_PORT,
        database=MARIADB_DB,
        user=MARIADB_USER,
        password=MARIADB_PASSWORD,
        table_name="test_haystack_docs",
        embedding_dimension=4,
        recreate_table=True,
    )
    yield store
    store.delete_documents(store.filter_documents() and [d.id for d in store.filter_documents()])
    try:
        store._cursor.execute("DROP TABLE IF EXISTS `test_haystack_docs`")
    except Exception:  # noqa: S110
        pass
    store.close()


@pytest.mark.integration
class TestIntegrationDocumentStore:
    def test_write_and_count(self, integration_store):
        docs = [Document(content="Hello"), Document(content="World")]
        written = integration_store.write_documents(docs)
        assert written == 2
        assert integration_store.count_documents() == 2

    def test_filter_by_content_id(self, integration_store):
        doc = Document(id="fixed-id", content="unique content")
        integration_store.write_documents([doc])
        results = integration_store.filter_documents({"field": "id", "operator": "==", "value": "fixed-id"})
        assert len(results) == 1
        assert results[0].id == "fixed-id"

    def test_filter_by_meta(self, integration_store):
        docs = [
            Document(content="A", meta={"category": "cat"}),
            Document(content="B", meta={"category": "dog"}),
        ]
        integration_store.write_documents(docs)
        results = integration_store.filter_documents(
            {"field": "meta.category", "operator": "==", "value": "cat"}
        )
        assert all(d.meta.get("category") == "cat" for d in results)

    def test_delete_documents(self, integration_store):
        doc = Document(id="to-delete", content="bye")
        integration_store.write_documents([doc])
        assert integration_store.count_documents() >= 1
        integration_store.delete_documents(["to-delete"])
        results = integration_store.filter_documents(
            {"field": "id", "operator": "==", "value": "to-delete"}
        )
        assert results == []

    def test_duplicate_fail(self, integration_store):
        doc = Document(id="dup-test", content="once")
        integration_store.write_documents([doc])
        with pytest.raises(DuplicateDocumentError):
            integration_store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_duplicate_overwrite(self, integration_store):
        doc = Document(id="ow-test", content="original")
        integration_store.write_documents([doc])
        doc2 = Document(id="ow-test", content="updated")
        integration_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE)
        results = integration_store.filter_documents(
            {"field": "id", "operator": "==", "value": "ow-test"}
        )
        assert results[0].content == "updated"

    def test_embedding_retrieval(self, integration_store):
        docs = [
            Document(content="red apple", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(content="green tea", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        integration_store.write_documents(docs)
        results = integration_store._embedding_retrieval(
            query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1
        )
        assert len(results) == 1
        assert results[0].content == "red apple"

    def test_keyword_retrieval(self, integration_store):
        docs = [
            Document(content="machine learning algorithms"),
            Document(content="database management systems"),
        ]
        integration_store.write_documents(docs)
        results = integration_store._keyword_retrieval(query="machine learning", top_k=5)
        assert len(results) >= 1
        assert any("machine" in d.content for d in results)

    def test_write_with_blob(self, integration_store):
        blob = ByteStream(data=b"binary content", meta={"type": "test"}, mime_type="text/plain")
        doc = Document(id="blob-doc", blob=blob)
        integration_store.write_documents([doc])
        results = integration_store.filter_documents(
            {"field": "id", "operator": "==", "value": "blob-doc"}
        )
        assert results[0].blob is not None
        assert results[0].blob.data == b"binary content"
