# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from unittest.mock import MagicMock

import oracledb as _oracledb
import pytest
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.oracle import OracleConnectionConfig, OracleDocumentStore

_ORACLE_FEATURE_INTEGRATION_FILE = "test_oracle_features_integration.py"
_ORACLE_FEATURE_SKIP_REASONS = {
    "ORA-00904": "Oracle vector APIs are unavailable in this live database",
    "ORA-51962": "Oracle vector memory area is exhausted in this live database",
}


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _oracle_feature_skip_reason(exc: BaseException) -> str | None:
    if not isinstance(exc, _oracledb.DatabaseError):
        return None
    message = str(exc)
    for error_code, reason in _ORACLE_FEATURE_SKIP_REASONS.items():
        if error_code in message:
            return reason
    if "PLS-00201" in message and "DBMS_VECTOR_CHAIN" in message:
        return "Oracle DBMS_VECTOR_CHAIN APIs are unavailable in this live database"
    return None


def _is_oracle_feature_integration_test(item) -> bool:
    return item.path.name == _ORACLE_FEATURE_INTEGRATION_FILE and item.get_closest_marker("integration") is not None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or not report.failed or call.excinfo is None:
        return
    if not _is_oracle_feature_integration_test(item):
        return

    reason = _oracle_feature_skip_reason(call.excinfo.value)
    if reason is None:
        return

    report.outcome = "skipped"
    report.longrepr = (str(item.path), report.location[1], f"Skipped: {reason}")


def connection_config(*, secret_source: str = "token") -> OracleConnectionConfig:
    wallet_location = _env_value("ORACLE_WALLET_LOCATION")
    if secret_source == "env_var":
        wallet_password = Secret.from_env_var("ORACLE_WALLET_PASSWORD", strict=False) if wallet_location else None
        return OracleConnectionConfig(
            user=Secret.from_env_var("ORACLE_USER", strict=False),
            password=Secret.from_env_var("ORACLE_PASSWORD", strict=False),
            dsn=Secret.from_env_var("ORACLE_DSN", strict=False),
            wallet_location=wallet_location,
            wallet_password=wallet_password,
        )

    wallet_password = _env_value("ORACLE_WALLET_PASSWORD")
    return OracleConnectionConfig(
        user=Secret.from_token(_env_value("ORACLE_USER", default="haystack")),
        password=Secret.from_token(_env_value("ORACLE_PASSWORD", default="haystack")),
        dsn=Secret.from_token(_env_value("ORACLE_DSN", default="localhost:1521/freepdb1")),
        wallet_location=wallet_location,
        wallet_password=Secret.from_token(wallet_password) if wallet_password else None,
    )


@pytest.fixture(name="connection_config")
def connection_config_fixture():
    return connection_config


def _make_store(table: str, embedding_dim: int) -> OracleDocumentStore:
    return OracleDocumentStore(
        connection_config=connection_config(),
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
    try:
        yield s
    finally:
        try:
            s.delete_table()
        finally:
            s.close()


@pytest.fixture
def embedding_store():
    """4-dim store for embedding-retrieval, HNSW, and async tests."""
    table = f"hs_emb_{uuid.uuid4().hex[:8]}"
    s = _make_store(table, embedding_dim=4)
    try:
        yield s
    finally:
        try:
            s.delete_table()
        finally:
            s.close()


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
    monkeypatch.delenv("ORACLE_WALLET_LOCATION", raising=False)
    monkeypatch.delenv("ORACLE_WALLET_PASSWORD", raising=False)
    return OracleDocumentStore(
        connection_config=connection_config(secret_source="env_var"),
        table_name="test_docs",
        embedding_dim=4,
        create_table_if_not_exists=False,
    )


@pytest.fixture
def mock_store():
    """MagicMock of OracleDocumentStore for retriever unit tests."""
    store = MagicMock(spec=OracleDocumentStore)
    store.table_name = "test_docs"
    store.distance_metric = "COSINE"
    store._embedding_retrieval.return_value = [Document(id="A" * 32, content="hi")]
    store._embedding_retrieval_async.return_value = [Document(id="A" * 32, content="hi")]
    store._keyword_retrieval.return_value = [Document(id="A" * 32, content="hi")]
    store._keyword_retrieval_async.return_value = [Document(id="A" * 32, content="hi")]
    store._hybrid_retrieval.return_value = [Document(id="A" * 32, content="hi")]
    store._hybrid_retrieval_async.return_value = [Document(id="A" * 32, content="hi")]
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
