# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace as dc_replace
from unittest.mock import MagicMock

import mariadb
import pytest
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
)
from haystack.utils import Secret

from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
from haystack_integrations.document_stores.mariadb.document_store import (
    _bytes_to_embedding,
    _document_to_row,
    _embedding_to_bytes,
    _rows_to_documents,
)


def _record(**kwargs) -> dict:
    base = {
        "id": "x",
        "content": "c",
        "embedding": None,
        "blob_data": None,
        "blob_meta": None,
        "blob_mime_type": None,
        "meta": "{}",
    }
    base.update(kwargs)
    return base


class TestSerialization:
    def test_to_dict(self):
        d = MariaDBDocumentStore().to_dict()
        assert d["type"] == "haystack_integrations.document_stores.mariadb.document_store.MariaDBDocumentStore"
        params = d["init_parameters"]
        assert params["host"] == "127.0.0.1"
        assert params["embedding_dimension"] == 768
        assert params["distance"] == "cosine"

    def test_from_dict_roundtrip(self):
        store = MariaDBDocumentStore(embedding_dimension=512, distance="euclidean")
        restored = MariaDBDocumentStore.from_dict(store.to_dict())
        assert restored.embedding_dimension == 512
        assert restored.host == "127.0.0.1"
        assert restored.distance == "euclidean"

    def test_invalid_table_name_raises(self):
        with pytest.raises(ValueError, match="table_name must contain only"):
            MariaDBDocumentStore(table_name="bad-table!")

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError, match="distance must be one of"):
            MariaDBDocumentStore(distance="manhattan")


class TestEmbeddingHelpers:
    def test_round_trip_embedding(self):
        original = [0.1, 0.2, 0.3, -0.5]
        recovered = _bytes_to_embedding(_embedding_to_bytes(original))
        assert len(recovered) == len(original)
        for a, b in zip(recovered, original, strict=True):
            assert abs(a - b) < 1e-6

    def test_embedding_bytes_length(self):
        assert len(_embedding_to_bytes([1.0] * 8)) == 8 * 4


class TestDocumentHelpers:
    def test_document_to_row_simple(self):
        row = _document_to_row(Document(id="abc", content="hello", meta={"x": 1}))
        assert row[0] == "abc"
        assert row[1] is None
        assert row[2] == "hello"
        assert row[3] is None
        assert row[6] == '{"x": 1}'

    def test_document_to_row_with_embedding(self):
        row = _document_to_row(Document(id="d1", content="text", embedding=[0.5, 0.5]))
        assert isinstance(row[1], bytes)

    def test_document_to_row_with_blob(self):
        blob = ByteStream(data=b"raw bytes", meta={"k": "v"}, mime_type="application/octet-stream")
        row = _document_to_row(Document(id="d2", content=None, blob=blob))
        assert row[3] == b"raw bytes"
        assert row[4] == '{"k": "v"}'
        assert row[5] == "application/octet-stream"

    def test_rows_to_documents_simple(self):
        docs = _rows_to_documents([_record(id="x1", content="text", meta='{"a": 1}')])
        assert len(docs) == 1
        assert docs[0].id == "x1"
        assert docs[0].meta == {"a": 1}

    def test_rows_to_documents_bytes_embedding(self):
        docs = _rows_to_documents([_record(id="e1", content="hi", embedding=_embedding_to_bytes([0.1, 0.9]))])
        assert docs[0].embedding is not None
        assert len(docs[0].embedding) == 2

    def test_rows_to_documents_with_blob(self):
        docs = _rows_to_documents(
            [_record(id="b1", content=None, blob_data=b"data", blob_meta='{"k": "v"}', blob_mime_type="image/png")]
        )
        assert docs[0].blob is not None
        assert docs[0].blob.data == b"data"
        assert docs[0].blob.mime_type == "image/png"

    def test_rows_to_documents_meta_none_vector_obj_and_dict_blob_meta(self):
        class FakeVec:
            def tolist(self):
                return [0.1, 0.2]

        records = [
            _record(id="v", content="c", embedding=FakeVec(), meta=None),
            _record(id="b", content=None, blob_data=b"d", blob_meta={"k": "v"}, blob_mime_type="image/png"),
        ]
        docs = _rows_to_documents(records)
        assert docs[0].embedding == [0.1, 0.2]
        assert docs[0].meta == {}
        assert docs[1].blob.meta == {"k": "v"}


class TestCountDocuments:
    @pytest.mark.parametrize(("row", "expected"), [({"cnt": 5}, 5), ({"cnt": 0}, 0), (None, 0)])
    def test_count_documents(self, mock_store, row, expected):
        store = mock_store()
        store._cursor.fetchone.return_value = row
        assert store.count_documents() == expected


class TestWriteDocuments:
    def test_write_empty_returns_zero(self, mock_store):
        assert mock_store().write_documents([]) == 0

    @pytest.mark.parametrize(
        ("policy", "sql_fragment"),
        [
            (DuplicatePolicy.NONE, "INSERT INTO"),
            (DuplicatePolicy.OVERWRITE, "ON DUPLICATE KEY UPDATE"),
            (DuplicatePolicy.SKIP, "INSERT IGNORE"),
        ],
    )
    def test_write_policy_selects_sql(self, mock_store, policy, sql_fragment):
        store = mock_store()
        store._cursor.rowcount = 1
        store.write_documents([Document(content="test")], policy=policy)
        assert sql_fragment in store._cursor.executemany.call_args[0][0]

    def test_write_fail_raises_on_duplicate(self, mock_store):
        store = mock_store()
        store._cursor.executemany.side_effect = mariadb.IntegrityError("Duplicate entry")
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([Document(id="dup", content="test")], policy=DuplicatePolicy.FAIL)

    def test_write_invalid_type_raises(self, mock_store):
        with pytest.raises(ValueError, match="list of Document objects"):
            mock_store().write_documents(["not a doc"])


class TestDeleteDocuments:
    def test_delete_empty_noop(self, mock_store):
        store = mock_store()
        store.delete_documents([])
        store._cursor.execute.assert_not_called()

    def test_delete_calls_execute(self, mock_store):
        store = mock_store()
        store.delete_documents(["id1", "id2"])
        sql = store._cursor.execute.call_args[0][0]
        assert "DELETE FROM" in sql
        assert "IN (?, ?)" in sql


class TestFilterDocuments:
    @pytest.mark.parametrize(
        ("filters", "where_present"),
        [(None, False), ({"field": "meta.x", "operator": "==", "value": 1}, True)],
    )
    def test_filter_where_clause(self, mock_store, filters, where_present):
        store = mock_store()
        store._cursor.fetchall.return_value = []
        store.filter_documents(filters=filters)
        assert ("WHERE" in store._cursor.execute.call_args[0][0]) == where_present


class TestEmbeddingRetrieval:
    def test_dimension_mismatch_raises(self, mock_store):
        store = mock_store(embedding_dimension=4)
        with pytest.raises(ValueError, match="query_embedding has 3 dimensions"):
            store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3])

    def test_filters_and_score_threshold(self, mock_store):
        store = mock_store(embedding_dimension=2)
        store._cursor.fetchall.return_value = [
            _record(id="keep", content="keep", score=0.1),
            _record(id="drop", content="drop", score=0.8),
        ]
        results = store._embedding_retrieval(
            query_embedding=[0.1, 0.2],
            filters={"field": "meta.x", "operator": "==", "value": 1},
            score_threshold=0.5,
        )
        assert [d.id for d in results] == ["keep"]
        assert "AND" in store._cursor.execute.call_args[0][0]


class TestKeywordRetrieval:
    def test_with_filters(self, mock_store):
        store = mock_store()
        store._cursor.fetchall.return_value = [_record(id="k", content="hit", score=1.2)]
        results = store._keyword_retrieval(query="hi", filters={"field": "meta.x", "operator": "==", "value": 1})
        assert results[0].id == "k"
        assert "WHERE" in store._cursor.execute.call_args[0][0]


class TestConnectionAndErrors:
    @pytest.mark.parametrize("op", ["write", "delete", "create", "drop"])
    def test_db_error_wrapped(self, mock_store, op):
        store = mock_store()
        store._cursor.executemany.side_effect = mariadb.Error("boom")
        store._cursor.execute.side_effect = mariadb.Error("boom")
        with pytest.raises(DocumentStoreError):
            if op == "write":
                store.write_documents([Document(content="x")])
            elif op == "delete":
                store.delete_documents(["a"])
            elif op == "create":
                store._initialize_table()
            else:
                store._drop_table()

    def test_connect_failure_wrapped(self, monkeypatch):
        monkeypatch.setattr(mariadb, "connect", MagicMock(side_effect=mariadb.Error("no")))
        store = MariaDBDocumentStore(user=Secret.from_token("u"), password=Secret.from_token("p"))
        with pytest.raises(DocumentStoreError):
            store._ensure_connection()

    def test_create_vector_index_ddl(self, mock_store):
        store = mock_store(create_vector_index=True)
        store._table_initialized = False
        store._initialize_table()
        assert "VECTOR INDEX" in store._cursor.execute.call_args[0][0]

    def test_close_swallows_errors(self, mock_store):
        store = mock_store()
        store._cursor.close.side_effect = Exception("x")
        store._connection.close.side_effect = Exception("y")
        store.close()
        assert store._connection is None


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        assert len(received) == len(expected)
        received.sort(key=lambda d: d.id)
        expected.sort(key=lambda d: d.id)
        for r, e in zip(received, expected, strict=True):
            if r.embedding is None:
                assert e.embedding is None
            else:
                assert r.embedding == pytest.approx(e.embedding)
            assert dc_replace(r, embedding=None) == dc_replace(e, embedding=None)

    def test_write_documents(self, document_store):
        docs = [Document(id="1", content="test")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_with_blob(self, document_store):
        blob = ByteStream(data=b"binary content", meta={"type": "test"}, mime_type="text/plain")
        doc = Document(id="blob-doc", blob=blob)
        document_store.write_documents([doc])
        results = document_store.filter_documents({"field": "id", "operator": "==", "value": "blob-doc"})
        assert results[0].blob is not None
        assert results[0].blob.data == b"binary content"
