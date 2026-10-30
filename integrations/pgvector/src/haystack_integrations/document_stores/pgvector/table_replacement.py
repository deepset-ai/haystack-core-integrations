# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Safe transactional table replacement helpers for PgvectorDocumentStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from haystack.document_stores.errors import DocumentStoreError
from psycopg import Error
from psycopg.sql import SQL, Identifier

if TYPE_CHECKING:
    from .document_store import PgvectorDocumentStore

_POSTGRES_IDENTIFIER_MAX_BYTES = 63
_DATABASE_IDENTITY_QUERY = SQL(
    "SELECT current_database(), (SELECT oid FROM pg_database WHERE datname = current_database()), "
    "inet_server_addr()::text, inet_server_port()"
)
_TABLE_EXISTS_QUERY = SQL(
    "SELECT c.relkind FROM pg_class AS c "
    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE n.nspname = %s AND c.relname = %s"
)
_COLUMNS_QUERY = SQL(
    "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), a.attnotnull, "
    "a.attidentity, a.attgenerated "
    "FROM pg_attribute AS a "
    "JOIN pg_class AS c ON c.oid = a.attrelid "
    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped "
    "ORDER BY a.attnum"
)
_CONSTRAINTS_QUERY = SQL(
    "SELECT con.contype, pg_catalog.pg_get_constraintdef(con.oid) "
    "FROM pg_constraint AS con "
    "JOIN pg_class AS c ON c.oid = con.conrelid "
    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE n.nspname = %s AND c.relname = %s "
    "ORDER BY con.contype, pg_catalog.pg_get_constraintdef(con.oid)"
)
_INDEX_EXISTS_QUERY = SQL("SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s")
_RELATION_EXISTS_QUERY = SQL(
    "SELECT 1 FROM pg_class AS c JOIN pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE n.nspname = %s AND c.relname = %s"
)


def _truncate_identifier(identifier: str) -> str:
    """Keep a generated PostgreSQL identifier within PostgreSQL's 63-byte limit."""
    encoded = identifier.encode("utf-8")
    if len(encoded) <= _POSTGRES_IDENTIFIER_MAX_BYTES:
        return identifier

    suffix = f"_{uuid4().hex[:8]}"
    prefix_bytes = encoded[: _POSTGRES_IDENTIFIER_MAX_BYTES - len(suffix)]
    return prefix_bytes.decode("utf-8", errors="ignore") + suffix


def _validate_identifier(identifier: str, label: str) -> None:
    if not identifier:
        msg = f"{label} must not be empty"
        raise ValueError(msg)
    if len(identifier.encode("utf-8")) > _POSTGRES_IDENTIFIER_MAX_BYTES:
        msg = f"{label} must be at most {_POSTGRES_IDENTIFIER_MAX_BYTES} UTF-8 bytes"
        raise ValueError(msg)


def _validate_stores(
    active_store: PgvectorDocumentStore,
    staging_store: PgvectorDocumentStore,
    backup_table_name: str | None,
    lock_timeout_ms: int,
) -> tuple[str, list[tuple[str, str, str]]]:
    if staging_store is active_store:
        msg = "staging_store must be a different PgvectorDocumentStore instance"
        raise ValueError(msg)
    if active_store.schema_name != staging_store.schema_name:
        msg = "Active and staging document stores must use the same schema"
        raise ValueError(msg)
    if active_store.table_name == staging_store.table_name:
        msg = "Active and staging document stores must use different table names"
        raise ValueError(msg)
    if isinstance(lock_timeout_ms, bool) or not isinstance(lock_timeout_ms, int) or lock_timeout_ms <= 0:
        msg = "lock_timeout_ms must be a positive integer"
        raise ValueError(msg)

    compatible_options = (
        "embedding_dimension",
        "vector_type",
        "vector_function",
        "language",
        "search_strategy",
    )
    for option in compatible_options:
        if getattr(active_store, option) != getattr(staging_store, option):
            msg = f"Active and staging document stores must use the same {option}"
            raise ValueError(msg)
    if active_store.search_strategy == "hnsw" and (
        active_store.hnsw_index_creation_kwargs != staging_store.hnsw_index_creation_kwargs
    ):
        msg = "Active and staging document stores must use the same hnsw_index_creation_kwargs"
        raise ValueError(msg)
    if active_store.keyword_index_name == staging_store.keyword_index_name:
        msg = "Active and staging keyword indexes must have different names"
        raise ValueError(msg)
    if active_store.search_strategy == "hnsw" and active_store.hnsw_index_name == staging_store.hnsw_index_name:
        msg = "Active and staging HNSW indexes must have different names"
        raise ValueError(msg)

    for name, label in (
        (active_store.schema_name, "schema_name"),
        (active_store.table_name, "active table_name"),
        (staging_store.table_name, "staging table_name"),
        (active_store.keyword_index_name, "active keyword_index_name"),
        (staging_store.keyword_index_name, "staging keyword_index_name"),
    ):
        _validate_identifier(name, label)
    if active_store.search_strategy == "hnsw":
        _validate_identifier(active_store.hnsw_index_name, "active hnsw_index_name")
        _validate_identifier(staging_store.hnsw_index_name, "staging hnsw_index_name")

    if backup_table_name is None:
        backup_table_name = _truncate_identifier(f"{active_store.table_name}_backup_{uuid4().hex[:8]}")
    else:
        _validate_identifier(backup_table_name, "backup_table_name")
    if backup_table_name in {active_store.table_name, staging_store.table_name}:
        msg = "backup_table_name must differ from the active and staging table names"
        raise ValueError(msg)

    token = uuid4().hex[:8]
    backup_keyword_index_name = _truncate_identifier(f"{backup_table_name}_kw_{token}")
    index_renames = [(active_store.keyword_index_name, staging_store.keyword_index_name, backup_keyword_index_name)]
    if active_store.search_strategy == "hnsw":
        backup_hnsw_index_name = _truncate_identifier(f"{backup_table_name}_hnsw_{token}")
        index_renames.append((active_store.hnsw_index_name, staging_store.hnsw_index_name, backup_hnsw_index_name))

    for _, _, backup_index_name in index_renames:
        if backup_index_name in {active_store.table_name, staging_store.table_name, backup_table_name}:
            msg = "Generated backup index name conflicts with a table name"
            raise ValueError(msg)
    return backup_table_name, index_renames


def _prepare_sync_store(store: PgvectorDocumentStore) -> None:
    """Open a connection without initializing or recreating a table."""
    recreate_table = store.recreate_table
    recreate_hnsw_index = store.hnsw_recreate_index_if_exists
    table_initialized = store._table_initialized
    store.recreate_table = False
    store.hnsw_recreate_index_if_exists = False
    # _ensure_db_setup initializes missing tables. A replacement must only promote a
    # pre-existing staging table, never silently create an empty one.
    store._table_initialized = True
    try:
        store._ensure_db_setup()
    finally:
        store.recreate_table = recreate_table
        store.hnsw_recreate_index_if_exists = recreate_hnsw_index
        store._table_initialized = table_initialized


async def _prepare_async_store(store: PgvectorDocumentStore) -> None:
    """Open an async connection without initializing or recreating a table."""
    recreate_table = store.recreate_table
    recreate_hnsw_index = store.hnsw_recreate_index_if_exists
    table_initialized = store._table_initialized
    store.recreate_table = False
    store.hnsw_recreate_index_if_exists = False
    store._table_initialized = True
    try:
        await store._ensure_db_setup_async()
    finally:
        store.recreate_table = recreate_table
        store.hnsw_recreate_index_if_exists = recreate_hnsw_index
        store._table_initialized = table_initialized


def _database_identity(connection: Any, cursor: Any) -> tuple[Any, ...]:
    row = cursor.execute(_DATABASE_IDENTITY_QUERY).fetchone()
    return (*row, connection.info.host, connection.info.port)


async def _database_identity_async(connection: Any, cursor: Any) -> tuple[Any, ...]:
    await cursor.execute(_DATABASE_IDENTITY_QUERY)
    row = await cursor.fetchone()
    return (*row, connection.info.host, connection.info.port)


def _relation_kind(cursor: Any, schema_name: str, table_name: str) -> str | None:
    row = cursor.execute(_TABLE_EXISTS_QUERY, (schema_name, table_name)).fetchone()
    return row[0] if row else None


async def _relation_kind_async(cursor: Any, schema_name: str, table_name: str) -> str | None:
    await cursor.execute(_TABLE_EXISTS_QUERY, (schema_name, table_name))
    row = await cursor.fetchone()
    return row[0] if row else None


def _columns(cursor: Any, schema_name: str, table_name: str) -> list[tuple[Any, ...]]:
    return cursor.execute(_COLUMNS_QUERY, (schema_name, table_name)).fetchall()


async def _columns_async(cursor: Any, schema_name: str, table_name: str) -> list[tuple[Any, ...]]:
    await cursor.execute(_COLUMNS_QUERY, (schema_name, table_name))
    return await cursor.fetchall()


def _constraints(cursor: Any, schema_name: str, table_name: str) -> list[tuple[Any, ...]]:
    return cursor.execute(_CONSTRAINTS_QUERY, (schema_name, table_name)).fetchall()


async def _constraints_async(cursor: Any, schema_name: str, table_name: str) -> list[tuple[Any, ...]]:
    await cursor.execute(_CONSTRAINTS_QUERY, (schema_name, table_name))
    return await cursor.fetchall()


def _has_index(cursor: Any, schema_name: str, table_name: str, index_name: str) -> bool:
    return bool(cursor.execute(_INDEX_EXISTS_QUERY, (schema_name, table_name, index_name)).fetchone())


async def _has_index_async(cursor: Any, schema_name: str, table_name: str, index_name: str) -> bool:
    await cursor.execute(_INDEX_EXISTS_QUERY, (schema_name, table_name, index_name))
    return bool(await cursor.fetchone())


def _relation_exists(cursor: Any, schema_name: str, relation_name: str) -> bool:
    return bool(cursor.execute(_RELATION_EXISTS_QUERY, (schema_name, relation_name)).fetchone())


async def _relation_exists_async(cursor: Any, schema_name: str, relation_name: str) -> bool:
    await cursor.execute(_RELATION_EXISTS_QUERY, (schema_name, relation_name))
    return bool(await cursor.fetchone())


def replace_table(
    active_store: PgvectorDocumentStore,
    staging_store: PgvectorDocumentStore,
    *,
    backup_table_name: str | None = None,
    lock_timeout_ms: int = 5000,
) -> str:
    """
    Atomically promote a staging table and retain the old active table as a backup.

    Writers must be paused for the cutover. The PostgreSQL transaction makes the table-name
    changes atomic, but it does not synchronize writes already in flight. Both stores must target
    the same database and schema and have compatible document-store/index configuration. Staging index names
    must be unique in the schema.

    :param active_store: The document store that currently serves queries.
    :param staging_store: A populated, compatible document store to promote.
    :param backup_table_name: Optional name to retain the former active table under.
    :param lock_timeout_ms: Maximum time in milliseconds to wait for table locks.
    :returns: The backup table name. The staging store instance is repointed to the backup.

    The backup is retained and ``recreate_table`` is disabled on both instances after success
    to prevent a later reconnect from unexpectedly dropping data.
    """
    backup_table_name, index_renames = _validate_stores(active_store, staging_store, backup_table_name, lock_timeout_ms)
    _prepare_sync_store(active_store)
    _prepare_sync_store(staging_store)
    assert active_store._connection is not None
    assert staging_store._connection is not None

    try:
        with active_store._connection.cursor() as active_cursor, staging_store._connection.cursor() as staging_cursor:
            if _database_identity(active_store._connection, active_cursor) != _database_identity(
                staging_store._connection, staging_cursor
            ):
                msg = "Active and staging document stores must connect to the same PostgreSQL database"
                raise ValueError(msg)

        schema_name = active_store.schema_name
        active_name = active_store.table_name
        staging_name = staging_store.table_name
        connection = active_store._connection

        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout_ms}ms",))

                if _relation_kind(cursor, schema_name, active_name) != "r":
                    msg = f"Active table {schema_name}.{active_name} does not exist as a regular table"
                    raise ValueError(msg)
                if _relation_kind(cursor, schema_name, staging_name) != "r":
                    msg = f"Staging table {schema_name}.{staging_name} does not exist as a regular table"
                    raise ValueError(msg)

                cursor.execute(
                    SQL("LOCK TABLE {schema}.{active}, {schema}.{staging} IN ACCESS EXCLUSIVE MODE").format(
                        schema=Identifier(schema_name),
                        active=Identifier(active_name),
                        staging=Identifier(staging_name),
                    )
                )

                if _columns(cursor, schema_name, active_name) != _columns(cursor, schema_name, staging_name):
                    msg = "Active and staging tables have incompatible column definitions"
                    raise ValueError(msg)
                if _constraints(cursor, schema_name, active_name) != _constraints(cursor, schema_name, staging_name):
                    msg = "Active and staging tables have incompatible constraints"
                    raise ValueError(msg)
                if _relation_exists(cursor, schema_name, backup_table_name):
                    msg = f"Backup relation {schema_name}.{backup_table_name} already exists"
                    raise ValueError(msg)

                for active_index, staging_index, backup_index in index_renames:
                    if not _has_index(cursor, schema_name, active_name, active_index):
                        msg = f"Expected index {active_index!r} was not found on active table {active_name!r}"
                        raise ValueError(msg)
                    if not _has_index(cursor, schema_name, staging_name, staging_index):
                        msg = f"Expected index {staging_index!r} was not found on staging table {staging_name!r}"
                        raise ValueError(msg)
                    if _relation_exists(cursor, schema_name, backup_index):
                        msg = f"Backup index relation {schema_name}.{backup_index} already exists"
                        raise ValueError(msg)

                for active_index, _, backup_index in index_renames:
                    cursor.execute(
                        SQL("ALTER INDEX {schema}.{old_name} RENAME TO {new_name}").format(
                            schema=Identifier(schema_name),
                            old_name=Identifier(active_index),
                            new_name=Identifier(backup_index),
                        )
                    )
                for active_index, staging_index, _ in index_renames:
                    cursor.execute(
                        SQL("ALTER INDEX {schema}.{old_name} RENAME TO {new_name}").format(
                            schema=Identifier(schema_name),
                            old_name=Identifier(staging_index),
                            new_name=Identifier(active_index),
                        )
                    )

                cursor.execute(
                    SQL("ALTER TABLE {schema}.{old_name} RENAME TO {new_name}").format(
                        schema=Identifier(schema_name),
                        old_name=Identifier(active_name),
                        new_name=Identifier(backup_table_name),
                    )
                )
                cursor.execute(
                    SQL("ALTER TABLE {schema}.{old_name} RENAME TO {new_name}").format(
                        schema=Identifier(schema_name),
                        old_name=Identifier(staging_name),
                        new_name=Identifier(active_name),
                    )
                )
    except Error as error:
        detailed_error_msg = (
            f"Could not atomically replace {active_store.schema_name}.{active_store.table_name} "
            f"with staging table {staging_store.schema_name}.{staging_store.table_name}: {error}"
        )
        raise DocumentStoreError(detailed_error_msg) from error

    active_store.recreate_table = False
    active_store.hnsw_recreate_index_if_exists = False
    active_store._table_initialized = True
    staging_store.table_name = backup_table_name
    staging_store.keyword_index_name = index_renames[0][2]
    if active_store.search_strategy == "hnsw":
        staging_store.hnsw_index_name = index_renames[1][2]
    staging_store.recreate_table = False
    staging_store.hnsw_recreate_index_if_exists = False
    staging_store._table_initialized = True
    return backup_table_name


async def replace_table_async(
    active_store: PgvectorDocumentStore,
    staging_store: PgvectorDocumentStore,
    *,
    backup_table_name: str | None = None,
    lock_timeout_ms: int = 5000,
) -> str:
    """
    Asynchronously promote a staging table and retain the old active table as a backup.

    Writers must be paused for the cutover. The transaction makes the table-name changes atomic,
    but it does not synchronize writes already in flight.

    :param active_store: The document store that currently serves queries.
    :param staging_store: A populated, compatible document store to promote.
    :param backup_table_name: Optional name to retain the former active table under.
    :param lock_timeout_ms: Maximum time in milliseconds to wait for table locks.
    :returns: The backup table name. The staging store instance is repointed to the backup.
    """
    backup_table_name, index_renames = _validate_stores(active_store, staging_store, backup_table_name, lock_timeout_ms)
    await _prepare_async_store(active_store)
    await _prepare_async_store(staging_store)
    assert active_store._async_connection is not None
    assert staging_store._async_connection is not None

    try:
        async with (
            active_store._async_connection.cursor() as active_cursor,
            staging_store._async_connection.cursor() as staging_cursor,
        ):
            active_identity = await _database_identity_async(active_store._async_connection, active_cursor)
            staging_identity = await _database_identity_async(staging_store._async_connection, staging_cursor)
            if active_identity != staging_identity:
                msg = "Active and staging document stores must connect to the same PostgreSQL database"
                raise ValueError(msg)

        schema_name = active_store.schema_name
        active_name = active_store.table_name
        staging_name = staging_store.table_name
        connection = active_store._async_connection

        async with connection.transaction():
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout_ms}ms",))

                if await _relation_kind_async(cursor, schema_name, active_name) != "r":
                    msg = f"Active table {schema_name}.{active_name} does not exist as a regular table"
                    raise ValueError(msg)
                if await _relation_kind_async(cursor, schema_name, staging_name) != "r":
                    msg = f"Staging table {schema_name}.{staging_name} does not exist as a regular table"
                    raise ValueError(msg)

                await cursor.execute(
                    SQL("LOCK TABLE {schema}.{active}, {schema}.{staging} IN ACCESS EXCLUSIVE MODE").format(
                        schema=Identifier(schema_name),
                        active=Identifier(active_name),
                        staging=Identifier(staging_name),
                    )
                )

                if await _columns_async(cursor, schema_name, active_name) != await _columns_async(
                    cursor, schema_name, staging_name
                ):
                    msg = "Active and staging tables have incompatible column definitions"
                    raise ValueError(msg)
                if await _constraints_async(cursor, schema_name, active_name) != await _constraints_async(
                    cursor, schema_name, staging_name
                ):
                    msg = "Active and staging tables have incompatible constraints"
                    raise ValueError(msg)
                if await _relation_exists_async(cursor, schema_name, backup_table_name):
                    msg = f"Backup relation {schema_name}.{backup_table_name} already exists"
                    raise ValueError(msg)

                for active_index, staging_index, backup_index in index_renames:
                    if not await _has_index_async(cursor, schema_name, active_name, active_index):
                        msg = f"Expected index {active_index!r} was not found on active table {active_name!r}"
                        raise ValueError(msg)
                    if not await _has_index_async(cursor, schema_name, staging_name, staging_index):
                        msg = f"Expected index {staging_index!r} was not found on staging table {staging_name!r}"
                        raise ValueError(msg)
                    if await _relation_exists_async(cursor, schema_name, backup_index):
                        msg = f"Backup index relation {schema_name}.{backup_index} already exists"
                        raise ValueError(msg)

                for active_index, _, backup_index in index_renames:
                    await cursor.execute(
                        SQL("ALTER INDEX {schema}.{old_name} RENAME TO {new_name}").format(
                            schema=Identifier(schema_name),
                            old_name=Identifier(active_index),
                            new_name=Identifier(backup_index),
                        )
                    )
                for active_index, staging_index, _ in index_renames:
                    await cursor.execute(
                        SQL("ALTER INDEX {schema}.{old_name} RENAME TO {new_name}").format(
                            schema=Identifier(schema_name),
                            old_name=Identifier(staging_index),
                            new_name=Identifier(active_index),
                        )
                    )

                await cursor.execute(
                    SQL("ALTER TABLE {schema}.{old_name} RENAME TO {new_name}").format(
                        schema=Identifier(schema_name),
                        old_name=Identifier(active_name),
                        new_name=Identifier(backup_table_name),
                    )
                )
                await cursor.execute(
                    SQL("ALTER TABLE {schema}.{old_name} RENAME TO {new_name}").format(
                        schema=Identifier(schema_name),
                        old_name=Identifier(staging_name),
                        new_name=Identifier(active_name),
                    )
                )
    except Error as error:
        detailed_error_msg = (
            f"Could not atomically replace {active_store.schema_name}.{active_store.table_name} "
            f"with staging table {staging_store.schema_name}.{staging_store.table_name}: {error}"
        )
        raise DocumentStoreError(detailed_error_msg) from error

    active_store.recreate_table = False
    active_store.hnsw_recreate_index_if_exists = False
    active_store._table_initialized = True
    staging_store.table_name = backup_table_name
    staging_store.keyword_index_name = index_renames[0][2]
    if active_store.search_strategy == "hnsw":
        staging_store.hnsw_index_name = index_renames[1][2]
    staging_store.recreate_table = False
    staging_store.hnsw_recreate_index_if_exists = False
    staging_store._table_initialized = True
    return backup_table_name
