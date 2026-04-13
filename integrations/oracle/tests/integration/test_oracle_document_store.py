"""Integration tests against a live Oracle 23ai instance.

Required environment variables:
    ORACLE_USER       — database username
    ORACLE_PASSWORD   — database password
    ORACLE_DSN        — e.g. localhost:1521/freepdb1

Optional (for ADB-S / wallet connections):
    ORACLE_WALLET_LOCATION
    ORACLE_WALLET_PASSWORD

Run with:
    pytest tests/integration/ -v
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.oracle import OracleConnectionConfig, OracleDocumentStore

pytestmark = pytest.mark.skipif(
    not os.getenv("ORACLE_DSN"),
    reason="ORACLE_DSN not set — skipping Oracle integration tests",
)


def _unique_table() -> str:
    return f"hs_test_{uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def store():
    table = _unique_table()
    s = OracleDocumentStore(
        connection_config=OracleConnectionConfig(
            user=os.environ["ORACLE_USER"],
            password=Secret.from_env_var("ORACLE_PASSWORD"),
            dsn=os.environ["ORACLE_DSN"],
            wallet_location=os.getenv("ORACLE_WALLET_LOCATION"),
            wallet_password=(
                Secret.from_env_var("ORACLE_WALLET_PASSWORD") if os.getenv("ORACLE_WALLET_PASSWORD") else None
            ),
        ),
        table_name=table,
        embedding_dim=4,
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )
    yield s
    # Teardown
    with s._get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE {table} PURGE")
        conn.commit()


def _docs(n: int = 3) -> list[Document]:
    return [
        Document(
            id=uuid4().hex.upper()[:32],
            content=f"document {i}",
            meta={"index": i, "lang": "en"},
            embedding=[float(i), float(i + 1), float(i + 2), float(i + 3)],
        )
        for i in range(n)
    ]


def test_create_table_idempotent(store):
    store._ensure_table()  # should not raise


def test_write_and_count(store):
    docs = _docs(3)
    store.write_documents(docs)
    assert store.count_documents() >= 3


def test_filter_documents_no_filter(store):
    store.write_documents(_docs(2))
    all_docs = store.filter_documents()
    assert len(all_docs) >= 2


def test_filter_documents_equality(store):
    unique_lang = f"lang_{uuid4().hex[:6]}"
    doc = Document(
        id=uuid4().hex.upper()[:32],
        content="unique lang doc",
        meta={"lang": unique_lang},
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    store.write_documents([doc])
    results = store.filter_documents(filters={"field": "meta.lang", "operator": "==", "value": unique_lang})
    assert len(results) == 1
    assert results[0].meta["lang"] == unique_lang


def test_filter_documents_in_operator(store):
    tag = uuid4().hex[:6]
    docs = [
        Document(id=uuid4().hex.upper()[:32], content="a", meta={"tag": f"{tag}_a"}, embedding=[1.0, 0.0, 0.0, 0.0]),
        Document(id=uuid4().hex.upper()[:32], content="b", meta={"tag": f"{tag}_b"}, embedding=[0.0, 1.0, 0.0, 0.0]),
        Document(id=uuid4().hex.upper()[:32], content="c", meta={"tag": f"{tag}_c"}, embedding=[0.0, 0.0, 1.0, 0.0]),
    ]
    store.write_documents(docs)
    results = store.filter_documents(filters={"field": "meta.tag", "operator": "in", "value": [f"{tag}_a", f"{tag}_b"]})
    assert len(results) == 2


def test_write_duplicate_none_policy_raises(store):
    from haystack.document_stores.errors import DuplicateDocumentError

    doc = Document(id=uuid4().hex.upper()[:32], content="dup", meta={}, embedding=[1.0, 0.0, 0.0, 0.0])
    store.write_documents([doc])
    with pytest.raises(DuplicateDocumentError):
        store.write_documents([doc], policy=DuplicatePolicy.NONE)


def test_write_duplicate_skip_policy_silently_ignores(store):
    doc = Document(id=uuid4().hex.upper()[:32], content="skip-me", meta={}, embedding=[1.0, 0.0, 0.0, 0.0])
    store.write_documents([doc])
    count_before = store.count_documents()
    store.write_documents([doc], policy=DuplicatePolicy.SKIP)
    assert store.count_documents() == count_before


def test_write_duplicate_overwrite_policy_updates_content(store):
    doc_id = uuid4().hex.upper()[:32]
    doc = Document(id=doc_id, content="original", meta={}, embedding=[1.0, 0.0, 0.0, 0.0])
    store.write_documents([doc])
    updated = Document(id=doc_id, content="updated", meta={}, embedding=[1.0, 0.0, 0.0, 0.0])
    store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)
    results = store.filter_documents(filters={"field": "id", "operator": "==", "value": doc_id})
    assert results[0].content == "updated"


def test_delete_documents(store):
    doc_id = uuid4().hex.upper()[:32]
    doc = Document(id=doc_id, content="to delete", meta={}, embedding=[1.0, 0.0, 0.0, 0.0])
    store.write_documents([doc])
    store.delete_documents([doc_id])
    results = store.filter_documents(filters={"field": "id", "operator": "==", "value": doc_id})
    assert len(results) == 0


def test_embedding_retrieval_returns_ordered_results(store):
    tag = uuid4().hex[:6]
    docs = [
        Document(id=uuid4().hex.upper()[:32], content="near", meta={"tag": tag}, embedding=[1.0, 0.0, 0.0, 0.0]),
        Document(id=uuid4().hex.upper()[:32], content="far", meta={"tag": tag}, embedding=[0.0, 0.0, 0.0, 1.0]),
        Document(id=uuid4().hex.upper()[:32], content="medium", meta={"tag": tag}, embedding=[0.7, 0.7, 0.0, 0.0]),
    ]
    store.write_documents(docs)
    results = store._embedding_retrieval(
        [1.0, 0.0, 0.0, 0.0],
        filters={"field": "meta.tag", "operator": "==", "value": tag},
        top_k=3,
    )
    assert results[0].content == "near"
    assert len(results) == 3


def test_embedding_retrieval_with_filter(store):
    tag = uuid4().hex[:6]
    docs = [
        Document(
            id=uuid4().hex.upper()[:32], content="en", meta={"tag": tag, "lang": "en"}, embedding=[1.0, 0.0, 0.0, 0.0]
        ),  # noqa: E501
        Document(
            id=uuid4().hex.upper()[:32], content="de", meta={"tag": tag, "lang": "de"}, embedding=[1.0, 0.0, 0.0, 0.0]
        ),  # noqa: E501
    ]
    store.write_documents(docs)
    results = store._embedding_retrieval(
        [1.0, 0.0, 0.0, 0.0],
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.tag", "operator": "==", "value": tag},
                {"field": "meta.lang", "operator": "==", "value": "en"},
            ],
        },
        top_k=10,
    )
    assert all(d.meta["lang"] == "en" for d in results)


def test_hnsw_index_creation(store):
    store.create_hnsw_index()
    with store._get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = :1",
            [f"{store.table_name.upper()}_VIDX"],
        )
        count = cur.fetchone()[0]
    assert count == 1


@pytest.mark.asyncio
async def test_async_write_and_retrieve(store):
    doc_id = uuid4().hex.upper()[:32]
    doc = Document(id=doc_id, content="async test", meta={}, embedding=[0.5, 0.5, 0.0, 0.0])
    await store.write_documents_async([doc])
    results = await store._embedding_retrieval_async([0.5, 0.5, 0.0, 0.0], top_k=1)
    assert len(results) >= 1
