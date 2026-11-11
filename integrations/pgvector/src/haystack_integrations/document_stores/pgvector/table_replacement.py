# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Safe transactional table replacement helpers for PgvectorDocumentStore."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from haystack.document_stores.errors import DocumentStoreError
from psycopg import AsyncCursor, Cursor, Error
from psycopg.sql import SQL, Composed, Identifier

if TYPE_CHECKING:
    from .document_store import PgvectorDocumentStore

_POSTGRES_IDENTIFIER_MAX_BYTES = 63
# Identifies "same database on the same running server" from the server side only, so it does not depend on
# how the client spelled the host (localhost vs 127.0.0.1, IPv4 vs IPv6, TCP vs Unix socket).
# pg_postmaster_start_time() is callable by any role, unlike pg_control_system().
_DATABASE_IDENTITY_QUERY = SQL(
    "SELECT current_database(), (SELECT oid FROM pg_database WHERE datname = current_database()), "
    "pg_postmaster_start_time()"
)
_TABLE_KINDS_QUERY = SQL(
    "SELECT c.relname, c.relkind FROM pg_class AS c "
    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE n.nspname = %s AND c.relname = ANY(%s::name[])"
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
_INDEXES_QUERY = SQL(
    "SELECT t.relname, i.relname, x.indisprimary FROM pg_index AS x "
    "JOIN pg_class AS t ON t.oid = x.indrelid "
    "JOIN pg_class AS i ON i.oid = x.indexrelid "
    "JOIN pg_namespace AS n ON n.oid = t.relnamespace "
    "WHERE n.nspname = %s AND t.relname = ANY(%s::name[])"
)
_EXISTING_RELATIONS_QUERY = SQL(
    "SELECT c.relname FROM pg_class AS c "
    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE n.nspname = %s AND c.relname = ANY(%s::name[])"
)
_RENAME_INDEX = "ALTER INDEX {schema}.{old_name} RENAME TO {new_name}"
_RENAME_TABLE = "ALTER TABLE {schema}.{old_name} RENAME TO {new_name}"


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
    if isinstance(lock_timeout_ms, bool) or not isinstance(lock_timeout_ms, int):
        msg = f"lock_timeout_ms must be an int, got {type(lock_timeout_ms).__name__}"
        raise TypeError(msg)
    if lock_timeout_ms <= 0:
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
    return backup_table_name, index_renames


@contextmanager
def _without_table_setup(store: PgvectorDocumentStore) -> Iterator[None]:
    """
    Let `store` connect without its first-connect table setup, then restore its settings.

    Setup creates a missing table and, with `recreate_table=True`, drops an existing one first. A replacement must
    neither promote a silently created empty table nor drop live data, so setup is skipped while connecting.
    """
    settings = (store.recreate_table, store.hnsw_recreate_index_if_exists, store._table_initialized)
    store.recreate_table = False
    store.hnsw_recreate_index_if_exists = False
    store._table_initialized = True
    try:
        yield
    finally:
        store.recreate_table, store.hnsw_recreate_index_if_exists, store._table_initialized = settings


def _prepare_sync_store(store: PgvectorDocumentStore) -> None:
    """Open a connection without initializing or recreating a table."""
    with _without_table_setup(store):
        store._ensure_db_setup()


async def _prepare_async_store(store: PgvectorDocumentStore) -> None:
    """Open an async connection without initializing or recreating a table."""
    with _without_table_setup(store):
        await store._ensure_db_setup_async()


_Row = tuple[Any, ...]


def _database_identity(cursor: Cursor) -> list[_Row]:
    return cursor.execute(_DATABASE_IDENTITY_QUERY).fetchall()


async def _database_identity_async(cursor: AsyncCursor) -> list[_Row]:
    await cursor.execute(_DATABASE_IDENTITY_QUERY)
    return await cursor.fetchall()


@dataclass(frozen=True)
class _Swap:
    """
    Names involved in one table replacement, with the SQL and checks to perform it.

    Holds no connection and performs no I/O, so the sync and async paths share every query, check, and
    statement and only differ in how they execute them.
    """

    schema: str
    active_table: str
    staging_table: str
    backup_table: str
    # (active name, staging name, backup name) for the keyword index and, with HNSW, the HNSW index.
    index_renames: tuple[tuple[str, str, str], ...]
    backup_pk: str
    lock_timeout_ms: int

    @classmethod
    def create(
        cls,
        active_store: PgvectorDocumentStore,
        staging_store: PgvectorDocumentStore,
        backup_table_name: str | None,
        lock_timeout_ms: int,
    ) -> _Swap:
        backup_table, index_renames = _validate_stores(active_store, staging_store, backup_table_name, lock_timeout_ms)
        return cls(
            schema=active_store.schema_name,
            active_table=active_store.table_name,
            staging_table=staging_store.table_name,
            backup_table=backup_table,
            index_renames=tuple(index_renames),
            backup_pk=_truncate_identifier(f"{backup_table}_pkey"),
            lock_timeout_ms=lock_timeout_ms,
        )

    def lock_timeout_query(self) -> tuple[SQL, tuple[str]]:
        return SQL("SELECT set_config('lock_timeout', %s, true)"), (f"{self.lock_timeout_ms}ms",)

    def table_kinds_query(self) -> tuple[SQL, tuple[str, list[str]]]:
        return _TABLE_KINDS_QUERY, (self.schema, [self.active_table, self.staging_table])

    def check_tables_exist(self, rows: list[_Row]) -> None:
        kinds: dict[str, str] = dict(rows)
        for label, table in (("Active", self.active_table), ("Staging", self.staging_table)):
            if kinds.get(table) != "r":
                msg = f"{label} table {self.schema}.{table} does not exist as a regular table"
                raise DocumentStoreError(msg)

    def lock_statement(self) -> Composed:
        return SQL("LOCK TABLE {schema}.{active}, {schema}.{staging} IN ACCESS EXCLUSIVE MODE").format(
            schema=Identifier(self.schema),
            active=Identifier(self.active_table),
            staging=Identifier(self.staging_table),
        )

    def catalog_queries(self) -> list[tuple[SQL, tuple[Any, ...]]]:
        """Queries whose results `plan` expects, in this order. Run them after taking the locks."""
        backup_relations = [self.backup_table, self.backup_pk, *(backup for _, _, backup in self.index_renames)]
        return [
            (_COLUMNS_QUERY, (self.schema, self.active_table)),
            (_COLUMNS_QUERY, (self.schema, self.staging_table)),
            (_CONSTRAINTS_QUERY, (self.schema, self.active_table)),
            (_CONSTRAINTS_QUERY, (self.schema, self.staging_table)),
            (_INDEXES_QUERY, (self.schema, [self.active_table, self.staging_table])),
            (_EXISTING_RELATIONS_QUERY, (self.schema, backup_relations)),
        ]

    def plan(self, results: list[list[_Row]]) -> list[Composed]:
        """
        Check the catalog results of `catalog_queries` and return the rename statements to execute, in order.

        :raises DocumentStoreError: If the tables are incompatible or a backup name is already taken.
        """
        active_columns, staging_columns, active_constraints, staging_constraints, index_rows, existing_rows = results
        if active_columns != staging_columns:
            msg = "Active and staging tables have incompatible column definitions"
            raise DocumentStoreError(msg)
        if active_constraints != staging_constraints:
            msg = "Active and staging tables have incompatible constraints"
            raise DocumentStoreError(msg)
        existing = {name for (name,) in existing_rows}
        if self.backup_table in existing:
            msg = f"Backup relation {self.schema}.{self.backup_table} already exists"
            raise DocumentStoreError(msg)

        indexes = {(table, index) for table, index, _ in index_rows}
        primary_keys = {table: index for table, index, is_primary in index_rows if is_primary}
        renames = list(self.index_renames)
        # PostgreSQL does not rename a table's primary key with the table, so without this the promoted table
        # keeps the staging name and names drift on every replacement. Renaming the index also renames its
        # constraint. Matching constraints guarantee both tables have a primary key or neither.
        if self.active_table in primary_keys and self.staging_table in primary_keys:
            renames.append((primary_keys[self.active_table], primary_keys[self.staging_table], self.backup_pk))

        for active_index, staging_index, backup_index in renames:
            if (self.active_table, active_index) not in indexes:
                msg = f"Expected index {active_index!r} was not found on active table {self.active_table!r}"
                raise DocumentStoreError(msg)
            if (self.staging_table, staging_index) not in indexes:
                msg = f"Expected index {staging_index!r} was not found on staging table {self.staging_table!r}"
                raise DocumentStoreError(msg)
            if backup_index in existing:
                msg = f"Backup index relation {self.schema}.{backup_index} already exists"
                raise DocumentStoreError(msg)

        # Active names are freed first, so the staging objects can take them over.
        return [
            *(self._rename(_RENAME_INDEX, active, backup) for active, _, backup in renames),
            *(self._rename(_RENAME_INDEX, staging, active) for active, staging, _ in renames),
            self._rename(_RENAME_TABLE, self.active_table, self.backup_table),
            self._rename(_RENAME_TABLE, self.staging_table, self.active_table),
        ]

    def _rename(self, template: str, old_name: str, new_name: str) -> Composed:
        return SQL(template).format(
            schema=Identifier(self.schema), old_name=Identifier(old_name), new_name=Identifier(new_name)
        )

    def error(self, error: Error) -> DocumentStoreError:
        msg = (
            f"Could not atomically replace {self.schema}.{self.active_table} "
            f"with staging table {self.schema}.{self.staging_table}: {error}"
        )
        return DocumentStoreError(msg)


def _point_stores_after_swap(
    active_store: PgvectorDocumentStore, staging_store: PgvectorDocumentStore, swap: _Swap
) -> None:
    """Repoint the staging instance at the backup and mark both tables as initialized."""
    staging_store.table_name = swap.backup_table
    staging_store.keyword_index_name = swap.index_renames[0][2]
    if active_store.search_strategy == "hnsw":
        staging_store.hnsw_index_name = swap.index_renames[1][2]
    # Both tables already exist. Without this, an instance that had not initialized its table before the swap
    # would run table initialization on its next reconnect, and `recreate_table=True` would drop the promoted table.
    active_store._table_initialized = True
    staging_store._table_initialized = True


def replace_table(
    active_store: PgvectorDocumentStore,
    staging_store: PgvectorDocumentStore,
    *,
    backup_table_name: str | None = None,
    lock_timeout_ms: int = 5000,
) -> str:
    """
    Replace the table of `active_store` with the table of `staging_store`.

    See `PgvectorDocumentStore.replace_table` for requirements, locking, and side effects.

    :param active_store: The document store that currently serves queries.
    :param staging_store: A populated document store whose table replaces the active table.
    :param backup_table_name: Name to keep the former table under, or `None` to generate one.
    :param lock_timeout_ms: How long to wait for the table locks, in milliseconds.
    :returns: The name of the backup table that now holds the former documents.
    """
    swap = _Swap.create(active_store, staging_store, backup_table_name, lock_timeout_ms)
    _prepare_sync_store(active_store)
    _prepare_sync_store(staging_store)
    assert active_store._connection is not None
    assert staging_store._connection is not None

    try:
        with active_store._connection.cursor() as active_cursor, staging_store._connection.cursor() as staging_cursor:
            if _database_identity(active_cursor) != _database_identity(staging_cursor):
                msg = "Active and staging document stores must connect to the same PostgreSQL database"
                raise ValueError(msg)

        connection = active_store._connection
        with connection.transaction(), connection.cursor() as cursor:
            cursor.execute(*swap.lock_timeout_query())
            swap.check_tables_exist(cursor.execute(*swap.table_kinds_query()).fetchall())
            cursor.execute(swap.lock_statement())
            results = [cursor.execute(query, params).fetchall() for query, params in swap.catalog_queries()]
            for statement in swap.plan(results):
                cursor.execute(statement)
    except Error as error:
        raise swap.error(error) from error

    _point_stores_after_swap(active_store, staging_store, swap)
    return swap.backup_table


async def replace_table_async(
    active_store: PgvectorDocumentStore,
    staging_store: PgvectorDocumentStore,
    *,
    backup_table_name: str | None = None,
    lock_timeout_ms: int = 5000,
) -> str:
    """
    Asynchronously replace the table of `active_store` with the table of `staging_store`.

    See `PgvectorDocumentStore.replace_table_async` for requirements, locking, and side effects.

    :param active_store: The document store that currently serves queries.
    :param staging_store: A populated document store whose table replaces the active table.
    :param backup_table_name: Name to keep the former table under, or `None` to generate one.
    :param lock_timeout_ms: How long to wait for the table locks, in milliseconds.
    :returns: The name of the backup table that now holds the former documents.
    """
    swap = _Swap.create(active_store, staging_store, backup_table_name, lock_timeout_ms)
    await _prepare_async_store(active_store)
    await _prepare_async_store(staging_store)
    assert active_store._async_connection is not None
    assert staging_store._async_connection is not None

    try:
        async with (
            active_store._async_connection.cursor() as active_cursor,
            staging_store._async_connection.cursor() as staging_cursor,
        ):
            if await _database_identity_async(active_cursor) != await _database_identity_async(staging_cursor):
                msg = "Active and staging document stores must connect to the same PostgreSQL database"
                raise ValueError(msg)

        connection = active_store._async_connection
        async with connection.transaction(), connection.cursor() as cursor:
            await cursor.execute(*swap.lock_timeout_query())
            await cursor.execute(*swap.table_kinds_query())
            swap.check_tables_exist(await cursor.fetchall())
            await cursor.execute(swap.lock_statement())
            results = []
            for query, params in swap.catalog_queries():
                await cursor.execute(query, params)
                results.append(await cursor.fetchall())
            for statement in swap.plan(results):
                await cursor.execute(statement)
    except Error as error:
        raise swap.error(error) from error

    _point_stores_after_swap(active_store, staging_store, swap)
    return swap.backup_table
