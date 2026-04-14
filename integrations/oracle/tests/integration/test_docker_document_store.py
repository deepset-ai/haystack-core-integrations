# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests against a local Oracle 23ai instance started via Docker.

Start the container before running:

    docker compose up -d
    # wait for the healthcheck to pass (~60 s on first boot)

Then run:

    hatch run test:integration

The store fixture creates a throw-away table for every test module and drops it
on teardown, so tests are fully isolated from any other state in the DB.
"""

from __future__ import annotations

import socket

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.oracle import OracleConnectionConfig, OracleDocumentStore

# ---------------------------------------------------------------------------
# Connection details matching docker-compose.yml
# ---------------------------------------------------------------------------
_USER = "haystack"
_PASSWORD = "haystack"
_DSN = "localhost:1521/freepdb1"
_TABLE = "hs_docker_test"
_DIM = 4


def _oracle_reachable() -> bool:
    """Return True if the local Oracle Docker container is accepting connections."""
    try:
        with socket.create_connection(("localhost", 1521), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _oracle_reachable(),
    reason="Local Oracle container not reachable on localhost:1521",
)


@pytest.fixture(scope="module")
def store():
    s = OracleDocumentStore(
        connection_config=OracleConnectionConfig(
            user=_USER,
            password=Secret.from_token(_PASSWORD),
            dsn=_DSN,
        ),
        table_name=_TABLE,
        embedding_dim=_DIM,
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )
    yield s
    with s._get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE {_TABLE} PURGE")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str, content: str = "hello", meta: dict | None = None, embedding: list[float] | None = None):
    return Document(
        id=doc_id,
        content=content,
        meta=meta or {},
        embedding=embedding or [1.0, 0.0, 0.0, 0.0],
    )


def _uid(suffix: str = "") -> str:
    """Return a 32-char uppercase hex ID, optionally tagged."""
    import uuid

    base = uuid.uuid4().hex.upper()[:28]
    return f"{base}{suffix.upper():>4}"[:32]


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_count_documents_empty(store):
    assert store.count_documents() >= 0  # table may have rows from other tests


# ---------------------------------------------------------------------------
# write_documents — NONE policy
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_write_documents_none_policy(store):
    doc = _doc(_uid())
    written = store.write_documents([doc], policy=DuplicatePolicy.NONE)
    assert written == 1


@pytest.mark.integration
def test_write_documents_none_policy_duplicate_raises(store):
    doc = _doc(_uid("A001"))
    store.write_documents([doc], policy=DuplicatePolicy.NONE)
    with pytest.raises(DuplicateDocumentError):
        store.write_documents([doc], policy=DuplicatePolicy.NONE)


@pytest.mark.integration
def test_write_documents_empty_list_returns_zero(store):
    assert store.write_documents([]) == 0


# ---------------------------------------------------------------------------
# write_documents — SKIP policy
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_write_documents_skip_ignores_duplicate(store):
    doc = _doc(_uid("B001"), content="original")
    store.write_documents([doc], policy=DuplicatePolicy.NONE)
    count_before = store.count_documents()
    written = store.write_documents([doc], policy=DuplicatePolicy.SKIP)
    assert written == 0
    assert store.count_documents() == count_before


@pytest.mark.integration
def test_write_documents_skip_returns_only_new(store):
    existing = _doc(_uid("C001"))
    new = _doc(_uid("C002"))
    store.write_documents([existing], policy=DuplicatePolicy.NONE)
    written = store.write_documents([existing, new], policy=DuplicatePolicy.SKIP)
    assert written == 1


# ---------------------------------------------------------------------------
# write_documents — OVERWRITE policy
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_write_documents_overwrite_updates_content(store):
    doc_id = _uid("D001")
    store.write_documents([_doc(doc_id, content="original")], policy=DuplicatePolicy.NONE)
    store.write_documents([_doc(doc_id, content="updated")], policy=DuplicatePolicy.OVERWRITE)
    results = store.filter_documents(filters={"field": "id", "operator": "==", "value": doc_id})
    assert results[0].content == "updated"


# ---------------------------------------------------------------------------
# filter_documents
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_filter_documents_no_filter_returns_all(store):
    store.write_documents([_doc(_uid("E001")), _doc(_uid("E002"))])
    docs = store.filter_documents()
    assert len(docs) >= 2


@pytest.mark.integration
def test_filter_documents_equality(store):
    tag = _uid("F001")
    doc = _doc(tag, meta={"tag": tag})
    store.write_documents([doc])
    results = store.filter_documents(filters={"field": "meta.tag", "operator": "==", "value": tag})
    assert len(results) == 1
    assert results[0].meta["tag"] == tag


@pytest.mark.integration
def test_filter_documents_in_operator(store):
    tag = _uid("G")[:8]
    docs = [
        _doc(_uid("G001"), meta={"tag": f"{tag}_a"}),
        _doc(_uid("G002"), meta={"tag": f"{tag}_b"}),
        _doc(_uid("G003"), meta={"tag": f"{tag}_c"}),
    ]
    store.write_documents(docs)
    results = store.filter_documents(filters={"field": "meta.tag", "operator": "in", "value": [f"{tag}_a", f"{tag}_b"]})
    assert len(results) == 2


@pytest.mark.integration
def test_filter_documents_numeric_gt(store):
    tag = _uid("H")[:8]
    docs = [
        _doc(_uid("H001"), meta={"tag": tag, "score": 10}),
        _doc(_uid("H002"), meta={"tag": tag, "score": 50}),
        _doc(_uid("H003"), meta={"tag": tag, "score": 90}),
    ]
    store.write_documents(docs)
    results = store.filter_documents(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.tag", "operator": "==", "value": tag},
                {"field": "meta.score", "operator": ">", "value": 40},
            ],
        }
    )
    assert all(d.meta["score"] > 40 for d in results)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_documents(store):
    doc_id = _uid("I001")
    store.write_documents([_doc(doc_id)])
    store.delete_documents([doc_id])
    results = store.filter_documents(filters={"field": "id", "operator": "==", "value": doc_id})
    assert len(results) == 0


@pytest.mark.integration
def test_delete_documents_empty_list_is_noop(store):
    count_before = store.count_documents()
    store.delete_documents([])
    assert store.count_documents() == count_before


# ---------------------------------------------------------------------------
# _embedding_retrieval
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_embedding_retrieval_order(store):
    tag = _uid("J")[:8]
    docs = [
        _doc(_uid("J001"), content="near", meta={"tag": tag}, embedding=[1.0, 0.0, 0.0, 0.0]),
        _doc(_uid("J002"), content="far", meta={"tag": tag}, embedding=[0.0, 0.0, 0.0, 1.0]),
        _doc(_uid("J003"), content="medium", meta={"tag": tag}, embedding=[0.7, 0.7, 0.0, 0.0]),
    ]
    store.write_documents(docs)
    results = store._embedding_retrieval(
        [1.0, 0.0, 0.0, 0.0],
        filters={"field": "meta.tag", "operator": "==", "value": tag},
        top_k=3,
    )
    assert results[0].content == "near"
    assert len(results) == 3


@pytest.mark.integration
def test_embedding_retrieval_top_k(store):
    tag = _uid("K")[:8]
    docs = [_doc(_uid(f"K{i:03}"), meta={"tag": tag}, embedding=[1.0, 0.0, 0.0, 0.0]) for i in range(5)]
    store.write_documents(docs)
    results = store._embedding_retrieval(
        [1.0, 0.0, 0.0, 0.0],
        filters={"field": "meta.tag", "operator": "==", "value": tag},
        top_k=3,
    )
    assert len(results) == 3


@pytest.mark.integration
def test_embedding_retrieval_with_filter(store):
    tag = _uid("L")[:8]
    docs = [
        _doc(_uid("L001"), content="en", meta={"tag": tag, "lang": "en"}, embedding=[1.0, 0.0, 0.0, 0.0]),
        _doc(_uid("L002"), content="de", meta={"tag": tag, "lang": "de"}, embedding=[1.0, 0.0, 0.0, 0.0]),
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
    assert len(results) == 1


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_write_and_count(store):
    doc = _doc(_uid("M001"))
    await store.write_documents_async([doc])
    count = await store.count_documents_async()
    assert count >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_filter_documents(store):
    tag = _uid("N")[:8]
    doc = _doc(_uid("N001"), meta={"tag": tag})
    await store.write_documents_async([doc])
    results = await store.filter_documents_async(filters={"field": "meta.tag", "operator": "==", "value": tag})
    assert len(results) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_delete_documents(store):
    doc_id = _uid("O001")
    await store.write_documents_async([_doc(doc_id)])
    await store.delete_documents_async([doc_id])
    results = await store.filter_documents_async(filters={"field": "id", "operator": "==", "value": doc_id})
    assert len(results) == 0
