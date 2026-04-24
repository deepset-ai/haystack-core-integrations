# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import uuid
from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.oracle import OracleConnectionConfig, OracleDocumentStore

_USER = "haystack"
_PASSWORD = "haystack"
_DSN = "localhost:1521/freepdb1"


def _make_store(table: str, embedding_dim: int) -> OracleDocumentStore:
    return OracleDocumentStore(
        connection_config=OracleConnectionConfig(
            user=Secret.from_token(_USER),
            password=Secret.from_token(_PASSWORD),
            dsn=Secret.from_token(_DSN),
        ),
        table_name=table,
        embedding_dim=embedding_dim,
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )


@pytest.fixture
def document_store():
    """768-dim store required by the mixin's filterable_docs fixture."""
    table = f"hs_sync_{uuid.uuid4().hex[:8]}"
    s = _make_store(table, embedding_dim=768)
    yield s
    with s._get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE {table} PURGE")
        conn.commit()


@pytest.fixture
def embedding_store():
    """4-dim store for embedding-retrieval, HNSW, and async tests."""
    table = f"hs_emb_{uuid.uuid4().hex[:8]}"
    s = _make_store(table, embedding_dim=4)
    yield s
    with s._get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE {table} PURGE")
        conn.commit()


@pytest.fixture
def mock_pool(monkeypatch):
    """Patch oracledb.create_pool to return a mock pool with a mock connection/cursor."""
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (0,)
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = lambda _: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value.__enter__ = lambda _: conn
    pool.acquire.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(
        "haystack_integrations.document_stores.oracle.document_store.oracledb.create_pool",
        lambda **_: pool,
    )
    return pool, conn, cursor


@pytest.fixture
def patched_store(monkeypatch):
    """Real OracleDocumentStore instance with a patched (mock) connection pool."""
    monkeypatch.setenv("ORACLE_USER", "u")
    monkeypatch.setenv("ORACLE_PASSWORD", "p")
    monkeypatch.setenv("ORACLE_DSN", "localhost/xe")
    return OracleDocumentStore(
        connection_config=OracleConnectionConfig(
            user=Secret.from_env_var("ORACLE_USER"),
            password=Secret.from_env_var("ORACLE_PASSWORD"),
            dsn=Secret.from_env_var("ORACLE_DSN"),
        ),
        table_name="test_docs",
        embedding_dim=4,
        create_table_if_not_exists=False,
    )


@pytest.fixture
def mock_store():
    """MagicMock of OracleDocumentStore for retriever unit tests."""
    store = MagicMock(spec=OracleDocumentStore)
    store.distance_metric = "COSINE"
    store._embedding_retrieval.return_value = [Document(id="A" * 32, content="hi")]
    store._embedding_retrieval_async.return_value = [Document(id="A" * 32, content="hi")]
    store._keyword_retrieval.return_value = [Document(id="A" * 32, content="hi")]
    store._keyword_retrieval_async.return_value = [Document(id="A" * 32, content="hi")]
    store.to_dict.return_value = {
        "type": "haystack_integrations.document_stores.oracle.document_store.OracleDocumentStore",
        "init_parameters": {
            "connection_config": {
                "user": {"type": "env_var", "env_vars": ["ORACLE_USER"], "strict": False},
                "password": {"type": "env_var", "env_vars": ["ORACLE_PASSWORD"], "strict": False},
                "dsn": {"type": "env_var", "env_vars": ["ORACLE_DSN"], "strict": False},
                "wallet_location": None,
                "wallet_password": None,
                "min_connections": 1,
                "max_connections": 5,
            },
            "table_name": "test_docs",
            "embedding_dim": 4,
            "distance_metric": "COSINE",
            "create_table_if_not_exists": False,
            "create_index": False,
            "hnsw_neighbors": 32,
            "hnsw_ef_construction": 200,
            "hnsw_accuracy": 95,
            "hnsw_parallel": 4,
        },
    }
    return store
