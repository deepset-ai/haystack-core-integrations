# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.document_stores.pgvector import document_store as document_store_module
from haystack_integrations.document_stores.pgvector.table_replacement import (
    _columns,
    _columns_async,
    _constraints,
    _constraints_async,
    _database_identity,
    _database_identity_async,
    _has_index,
    _has_index_async,
    _prepare_async_store,
    _prepare_sync_store,
    _relation_exists,
    _relation_exists_async,
    _relation_kind,
    _relation_kind_async,
    _truncate_identifier,
    _validate_identifier,
    _validate_stores,
)


def _make_store(
    table_name: str = "active_documents",
    keyword_index_name: str = "active_keyword_idx",
    hnsw_index_name: str = "active_hnsw_idx",
    **overrides: Any,
) -> PgvectorDocumentStore:
    options: dict[str, Any] = {
        "table_name": table_name,
        "keyword_index_name": keyword_index_name,
        "hnsw_index_name": hnsw_index_name,
    }
    options.update(overrides)
    return PgvectorDocumentStore(**options)


def _make_store_pair(
    *, active_options: dict[str, Any] | None = None, staging_options: dict[str, Any] | None = None
) -> tuple[PgvectorDocumentStore, PgvectorDocumentStore]:
    active = _make_store(**(active_options or {}))
    staging_defaults: dict[str, Any] = {
        "table_name": "staging_documents",
        "keyword_index_name": "staging_keyword_idx",
        "hnsw_index_name": "staging_hnsw_idx",
    }
    staging_defaults.update(staging_options or {})
    staging = _make_store(**staging_defaults)
    return active, staging


def test_truncate_identifier_keeps_identifiers_that_fit() -> None:
    identifier = "a" * 63

    assert _truncate_identifier(identifier) == identifier


def test_truncate_identifier_limits_generated_names_to_63_bytes() -> None:
    result = _truncate_identifier("a" * 100)

    assert len(result.encode("utf-8")) == 63
    assert result.startswith("a" * 54)
    assert len(result) == 63
    assert result[54] == "_"


def test_truncate_identifier_does_not_split_utf8_characters() -> None:
    result = _truncate_identifier("文" * 30)

    assert len(result.encode("utf-8")) <= 63
    assert result.encode("utf-8").decode("utf-8") == result


@pytest.mark.parametrize("identifier", ["", "a" * 64, "文" * 22])
def test_validate_identifier_rejects_empty_or_overlong_identifiers(identifier: str) -> None:
    with pytest.raises(ValueError, match=r"must not be empty|at most 63 UTF-8 bytes"):
        _validate_identifier(identifier, "test identifier")


def test_validate_identifier_accepts_identifier_at_byte_limit() -> None:
    _validate_identifier("a" * 63, "test identifier")


def test_validate_stores_generates_backup_table_and_keyword_index_names() -> None:
    active, staging = _make_store_pair()

    backup_name, renames = _validate_stores(active, staging, None, 5000)

    assert backup_name.startswith(f"{active.table_name}_backup_")
    assert len(backup_name.encode("utf-8")) <= 63
    assert renames[0][0] == active.keyword_index_name
    assert renames[0][1] == staging.keyword_index_name
    assert renames[0][2] not in {active.table_name, staging.table_name, backup_name}
    assert len(renames[0][2].encode("utf-8")) <= 63


def test_validate_stores_includes_hnsw_index_rename_when_enabled() -> None:
    active, staging = _make_store_pair(
        active_options={"search_strategy": "hnsw", "hnsw_index_creation_kwargs": {"m": 16}},
        staging_options={"search_strategy": "hnsw", "hnsw_index_creation_kwargs": {"m": 16}},
    )

    backup_name, renames = _validate_stores(active, staging, None, 5000)

    assert len(renames) == 2
    assert renames[1][0] == active.hnsw_index_name
    assert renames[1][1] == staging.hnsw_index_name
    assert len(renames[1][2].encode("utf-8")) <= 63
    assert backup_name not in {active.table_name, staging.table_name}


def test_validate_stores_rejects_same_store_instance() -> None:
    active, _ = _make_store_pair()

    with pytest.raises(ValueError, match="different PgvectorDocumentStore instance"):
        _validate_stores(active, active, None, 5000)


def test_validate_stores_rejects_different_schemas() -> None:
    active, staging = _make_store_pair(staging_options={"schema_name": "another_schema"})

    with pytest.raises(ValueError, match="same schema"):
        _validate_stores(active, staging, None, 5000)


def test_validate_stores_rejects_same_table_name() -> None:
    active = _make_store(table_name="shared", keyword_index_name="active_kw")
    staging = _make_store(table_name="shared", keyword_index_name="staging_kw")

    with pytest.raises(ValueError, match="different table names"):
        _validate_stores(active, staging, None, 5000)


@pytest.mark.parametrize("lock_timeout_ms", [True, False, 0, -1, 1.5, "1000", None])
def test_validate_stores_rejects_invalid_lock_timeouts(lock_timeout_ms: Any) -> None:
    active, staging = _make_store_pair()

    with pytest.raises(ValueError, match="positive integer"):
        _validate_stores(active, staging, None, lock_timeout_ms)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("embedding_dimension", 1024),
        ("vector_type", "halfvec"),
        ("vector_function", "inner_product"),
        ("language", "simple"),
        ("search_strategy", "hnsw"),
    ],
)
def test_validate_stores_rejects_incompatible_options(option: str, value: Any) -> None:
    active, staging = _make_store_pair(staging_options={option: value})

    with pytest.raises(ValueError, match=f"same {option}"):
        _validate_stores(active, staging, None, 5000)


def test_validate_stores_rejects_different_hnsw_creation_options() -> None:
    active, staging = _make_store_pair(
        active_options={"search_strategy": "hnsw", "hnsw_index_creation_kwargs": {"m": 16}},
        staging_options={"search_strategy": "hnsw", "hnsw_index_creation_kwargs": {"m": 32}},
    )

    with pytest.raises(ValueError, match="same hnsw_index_creation_kwargs"):
        _validate_stores(active, staging, None, 5000)


def test_validate_stores_rejects_duplicate_keyword_index_names() -> None:
    active, staging = _make_store_pair(staging_options={"keyword_index_name": "active_keyword_idx"})

    with pytest.raises(ValueError, match="keyword indexes must have different names"):
        _validate_stores(active, staging, None, 5000)


def test_validate_stores_rejects_duplicate_hnsw_index_names() -> None:
    active, staging = _make_store_pair(
        active_options={"search_strategy": "hnsw"},
        staging_options={"search_strategy": "hnsw", "hnsw_index_name": "active_hnsw_idx"},
    )

    with pytest.raises(ValueError, match="HNSW indexes must have different names"):
        _validate_stores(active, staging, None, 5000)


@pytest.mark.parametrize(
    ("active_options", "staging_options", "backup_table_name", "message"),
    [
        ({"schema_name": "s" * 64}, {"schema_name": "s" * 64}, None, "schema_name"),
        ({"table_name": "a" * 64}, None, None, "active table_name"),
        (None, {"table_name": "s" * 64}, None, "staging table_name"),
        ({"keyword_index_name": "k" * 64}, None, None, "active keyword_index_name"),
        (None, {"keyword_index_name": "k" * 64}, None, "staging keyword_index_name"),
        (None, None, "b" * 64, "backup_table_name"),
    ],
)
def test_validate_stores_rejects_overlong_identifiers(
    active_options: dict[str, Any] | None,
    staging_options: dict[str, Any] | None,
    backup_table_name: str | None,
    message: str,
) -> None:
    active, staging = _make_store_pair(active_options=active_options, staging_options=staging_options)

    with pytest.raises(ValueError, match=message):
        _validate_stores(active, staging, backup_table_name, 5000)


def test_validate_stores_rejects_overlong_hnsw_index_names() -> None:
    active, staging = _make_store_pair(
        active_options={"search_strategy": "hnsw", "hnsw_index_name": "a" * 64},
        staging_options={"search_strategy": "hnsw"},
    )

    with pytest.raises(ValueError, match="active hnsw_index_name"):
        _validate_stores(active, staging, None, 5000)


def test_validate_stores_rejects_backup_name_that_matches_existing_table() -> None:
    active, staging = _make_store_pair()

    for backup_name in (active.table_name, staging.table_name):
        with pytest.raises(ValueError, match="differ from the active and staging table names"):
            _validate_stores(active, staging, backup_name, 5000)


def test_prepare_sync_store_temporarily_disables_recreation_flags() -> None:
    store = Mock()
    store.recreate_table = True
    store.hnsw_recreate_index_if_exists = True
    store._table_initialized = False

    def ensure_setup() -> None:
        assert store.recreate_table is False
        assert store.hnsw_recreate_index_if_exists is False
        assert store._table_initialized is True

    store._ensure_db_setup.side_effect = ensure_setup
    _prepare_sync_store(store)

    store._ensure_db_setup.assert_called_once_with()
    assert store.recreate_table is True
    assert store.hnsw_recreate_index_if_exists is True
    assert store._table_initialized is False


def test_prepare_sync_store_restores_flags_when_setup_fails() -> None:
    store = Mock()
    store.recreate_table = True
    store.hnsw_recreate_index_if_exists = True
    store._table_initialized = False
    store._ensure_db_setup.side_effect = RuntimeError("connection failed")

    with pytest.raises(RuntimeError, match="connection failed"):
        _prepare_sync_store(store)

    assert store.recreate_table is True
    assert store.hnsw_recreate_index_if_exists is True
    assert store._table_initialized is False


@pytest.mark.asyncio
async def test_prepare_async_store_temporarily_disables_recreation_flags() -> None:
    store = Mock()
    store.recreate_table = True
    store.hnsw_recreate_index_if_exists = True
    store._table_initialized = False

    async def ensure_setup() -> None:
        assert store.recreate_table is False
        assert store.hnsw_recreate_index_if_exists is False
        assert store._table_initialized is True

    store._ensure_db_setup_async = AsyncMock(side_effect=ensure_setup)
    await _prepare_async_store(store)

    store._ensure_db_setup_async.assert_awaited_once_with()
    assert store.recreate_table is True
    assert store.hnsw_recreate_index_if_exists is True
    assert store._table_initialized is False


@pytest.mark.asyncio
async def test_prepare_async_store_restores_flags_when_setup_fails() -> None:
    store = Mock()
    store.recreate_table = True
    store.hnsw_recreate_index_if_exists = True
    store._table_initialized = False
    store._ensure_db_setup_async = AsyncMock(side_effect=RuntimeError("connection failed"))

    with pytest.raises(RuntimeError, match="connection failed"):
        await _prepare_async_store(store)

    assert store.recreate_table is True
    assert store.hnsw_recreate_index_if_exists is True
    assert store._table_initialized is False


def test_database_identity_includes_database_and_connection_endpoint() -> None:
    connection = Mock()
    connection.info.host = "localhost"
    connection.info.port = 5432
    cursor = Mock()
    cursor.execute.return_value = cursor
    cursor.fetchone.return_value = ("test_db", 1234, "127.0.0.1", 5432)

    identity = _database_identity(connection, cursor)

    assert identity == ("test_db", 1234, "127.0.0.1", 5432, "localhost", 5432)


@pytest.mark.asyncio
async def test_database_identity_async_includes_database_and_connection_endpoint() -> None:
    connection = Mock()
    connection.info.host = "localhost"
    connection.info.port = 5432
    cursor = Mock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=("test_db", 1234, "127.0.0.1", 5432))

    identity = await _database_identity_async(connection, cursor)

    assert identity == ("test_db", 1234, "127.0.0.1", 5432, "localhost", 5432)


def test_sync_catalog_helpers_delegate_to_cursor() -> None:
    cursor = Mock()
    cursor.execute.return_value = cursor
    cursor.fetchone.side_effect = [("r",), (1,), (1,)]
    cursor.fetchall.side_effect = [[("id", "text")], [("p", "PRIMARY KEY (id)")]]

    assert _relation_kind(cursor, "public", "documents") == "r"
    assert _columns(cursor, "public", "documents") == [("id", "text")]
    assert _constraints(cursor, "public", "documents") == [("p", "PRIMARY KEY (id)")]
    assert _has_index(cursor, "public", "documents", "documents_idx") is True
    assert _relation_exists(cursor, "public", "documents_backup") is True
    assert cursor.execute.call_count == 5


@pytest.mark.asyncio
async def test_async_catalog_helpers_delegate_to_cursor() -> None:
    cursor = Mock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(side_effect=[("r",), (1,), (1,)])
    cursor.fetchall = AsyncMock(side_effect=[[("id", "text")], [("p", "PRIMARY KEY (id)")]])

    assert await _relation_kind_async(cursor, "public", "documents") == "r"
    assert await _columns_async(cursor, "public", "documents") == [("id", "text")]
    assert await _constraints_async(cursor, "public", "documents") == [("p", "PRIMARY KEY (id)")]
    assert await _has_index_async(cursor, "public", "documents", "documents_idx") is True
    assert await _relation_exists_async(cursor, "public", "documents_backup") is True
    assert cursor.execute.await_count == 5


def test_document_store_replace_table_delegates_to_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    active, staging = _make_store_pair()
    replacement = Mock(return_value="active_backup")
    monkeypatch.setattr(document_store_module, "replace_table", replacement)

    result = active.replace_table(staging, backup_table_name="active_backup", lock_timeout_ms=1000)

    assert result == "active_backup"
    replacement.assert_called_once_with(active, staging, backup_table_name="active_backup", lock_timeout_ms=1000)


@pytest.mark.asyncio
async def test_document_store_replace_table_async_delegates_to_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    active, staging = _make_store_pair()
    replacement = AsyncMock(return_value="active_backup")
    monkeypatch.setattr(document_store_module, "replace_table_async", replacement)

    result = await active.replace_table_async(staging, backup_table_name="active_backup", lock_timeout_ms=1000)

    assert result == "active_backup"
    replacement.assert_awaited_once_with(active, staging, backup_table_name="active_backup", lock_timeout_ms=1000)
