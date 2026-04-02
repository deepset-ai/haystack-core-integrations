from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.oracle import OracleConnectionConfig, OracleDocumentStore


@pytest.fixture()
def mock_pool(monkeypatch):
    """Patch oracledb.create_pool to return a mock pool with a mock connection/cursor."""
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (0,)
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value.__enter__ = lambda s: conn
    pool.acquire.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(
        "haystack_integrations.document_stores.oracle.document_store.oracledb.create_pool",
        lambda **kw: pool,
    )
    return pool, conn, cursor


@pytest.fixture()
def store(mock_pool, monkeypatch):
    monkeypatch.setenv("ORACLE_PASSWORD", "p")
    return OracleDocumentStore(
        connection_config=OracleConnectionConfig(
            user="u",
            password=Secret.from_env_var("ORACLE_PASSWORD"),
            dsn="localhost/xe",
        ),
        table_name="test_docs",
        embedding_dim=4,
        create_table_if_not_exists=False,
    )


def _doc(content="hello", embedding=None, doc_id="AABB" * 8):
    return Document(id=doc_id, content=content, meta={"k": "v"}, embedding=embedding)


# ------------------------------------------------------------------
# write_documents
# ------------------------------------------------------------------


def test_write_documents_none_policy_calls_insert(store, mock_pool):
    _, _, cursor = mock_pool
    store.write_documents([_doc()], policy=DuplicatePolicy.NONE)
    cursor.executemany.assert_called_once()
    sql = cursor.executemany.call_args[0][0]
    assert "INSERT INTO" in sql
    assert ":doc_id" in sql


def test_write_documents_none_policy_duplicate_raises(store, mock_pool):
    import oracledb as _oracledb

    _, _, cursor = mock_pool
    cursor.executemany.side_effect = _oracledb.IntegrityError("ORA-00001")
    with pytest.raises(DuplicateDocumentError):
        store.write_documents([_doc()], policy=DuplicatePolicy.NONE)


def test_write_documents_skip_policy_uses_merge_not_matched(store, mock_pool):
    _, _, cursor = mock_pool
    store.write_documents([_doc()], policy=DuplicatePolicy.SKIP)
    sql = cursor.executemany.call_args[0][0]
    assert "MERGE INTO" in sql
    assert "WHEN NOT MATCHED" in sql
    assert "WHEN MATCHED" not in sql


def test_write_documents_overwrite_policy_uses_full_merge(store, mock_pool):
    _, _, cursor = mock_pool
    store.write_documents([_doc()], policy=DuplicatePolicy.OVERWRITE)
    sql = cursor.executemany.call_args[0][0]
    assert "MERGE INTO" in sql
    assert "WHEN MATCHED" in sql
    assert "WHEN NOT MATCHED" in sql


def test_write_documents_returns_count(store, mock_pool):
    count = store.write_documents([_doc(), _doc(doc_id="CCDD" * 8)], policy=DuplicatePolicy.NONE)
    assert count == 2


def test_write_documents_empty_list_returns_zero(store, mock_pool):
    _, _, cursor = mock_pool
    count = store.write_documents([], policy=DuplicatePolicy.NONE)
    assert count == 0
    cursor.executemany.assert_not_called()


# ------------------------------------------------------------------
# filter_documents
# ------------------------------------------------------------------


def test_filter_documents_no_filter_fetches_all(store, mock_pool):
    _, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("AABB" * 8, "hello", '{"k": "v"}'),
        ("CCDD" * 8, "world", "{}"),
    ]
    docs = store.filter_documents()
    assert len(docs) == 2
    sql = cursor.execute.call_args[0][0]
    assert "WHERE" not in sql


def test_filter_documents_equality_filter_produces_correct_sql(store, mock_pool):
    _, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    store.filter_documents(filters={"field": "meta.author", "operator": "==", "value": "Alice"})
    sql, params = cursor.execute.call_args[0]
    assert "JSON_VALUE(metadata, '$.author') = :p0" in sql
    assert params["p0"] == "Alice"


def test_filter_documents_and_filter(store, mock_pool):
    _, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    store.filter_documents(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.lang", "operator": "==", "value": "en"},
                {"field": "meta.year", "operator": ">", "value": 2020},
            ],
        }
    )
    sql, params = cursor.execute.call_args[0]
    assert "AND" in sql
    assert len(params) == 2


# ------------------------------------------------------------------
# delete_documents
# ------------------------------------------------------------------


def test_delete_documents_builds_correct_sql(store, mock_pool):
    _, _, cursor = mock_pool
    store.delete_documents(["AABB" * 8, "CCDD" * 8])
    sql = cursor.execute.call_args[0][0]
    assert "DELETE FROM" in sql
    assert "IN (:p0, :p1)" in sql


def test_delete_documents_empty_list_is_noop(store, mock_pool):
    _, _, cursor = mock_pool
    store.delete_documents([])
    cursor.execute.assert_not_called()


# ------------------------------------------------------------------
# count_documents
# ------------------------------------------------------------------


def test_count_documents_returns_value(store, mock_pool):
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (42,)
    assert store.count_documents() == 42


# ------------------------------------------------------------------
# serialization
# ------------------------------------------------------------------


def test_to_dict_does_not_expose_plain_password(store):
    d = store.to_dict()
    pw = d["init_parameters"]["connection_config"]["password"]
    # Secret serializes as {"type": "...", ...} — never a plain string
    assert isinstance(pw, dict)
    assert pw.get("type") == "env_var"  # stored as env-var reference, not plain token


def test_from_dict_roundtrip(store):
    d = store.to_dict()
    restored = OracleDocumentStore.from_dict(d)
    assert restored.table_name == store.table_name
    assert restored.embedding_dim == store.embedding_dim
    assert restored.distance_metric == store.distance_metric


# ------------------------------------------------------------------
# HNSW index SQL shape
# ------------------------------------------------------------------


def test_create_hnsw_index_sql(store, mock_pool):
    _, _, cursor = mock_pool
    store.create_hnsw_index()
    sql = cursor.execute.call_args[0][0]
    assert "CREATE VECTOR INDEX" in sql
    assert "HNSW" in sql
    assert str(store.hnsw_neighbors) in sql
    assert str(store.hnsw_ef_construction) in sql
