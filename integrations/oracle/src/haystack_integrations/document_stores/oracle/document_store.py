# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import array as _array
import asyncio
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal

import oracledb
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .filters import FilterTranslator

logger = logging.getLogger(__name__)

_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#]{0,127}$")
_SAFE_FIELD_PATH = re.compile(r"^[A-Za-z0-9_.]+$")
MAX_INDEX_NAME_LEN = 128


def _validate_field_path(field_path: str) -> None:
    if not _SAFE_FIELD_PATH.match(field_path):
        msg = f"Invalid metadata field name: {field_path!r}"
        raise ValueError(msg)


def _try_parse_number(value: Any) -> Any:
    """
    Attempt to parse a string as a number.

    Returns int for whole numbers, float for decimals, or the
    original value when conversion is not possible.
    """
    if value is None:
        return None
    try:
        f = float(value)
        i = int(f)
        return i if f == i else f
    except (ValueError, TypeError):
        return value


@dataclass
class OracleConnectionConfig:
    """
    Connection parameters for Oracle Database.

    Supports both thin (direct TCP) and thick (wallet / ADB-S) modes.
    Thin mode requires no Oracle Instant Client; thick mode is activated
    automatically when *wallet_location* is provided.
    """

    user: Secret
    password: Secret
    dsn: Secret
    wallet_location: str | None = None
    wallet_password: Secret | None = None
    min_connections: int = 1
    max_connections: int = 5

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return {
            "user": self.user.to_dict(),
            "password": self.password.to_dict(),
            "dsn": self.dsn.to_dict(),
            "wallet_location": self.wallet_location,
            "wallet_password": self.wallet_password.to_dict() if self.wallet_password else None,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleConnectionConfig":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data, keys=["user", "password", "dsn", "wallet_password"])
        return cls(**data)


class OracleDocumentStore:
    """
    Haystack DocumentStore backed by Oracle AI Vector Search.

    Requires Oracle Database 23ai or later (for VECTOR data type and
    IF NOT EXISTS DDL support).

    Usage::

        from haystack.utils import Secret
        from haystack_integrations.document_stores.oracle import (
            OracleDocumentStore, OracleConnectionConfig,
        )

        store = OracleDocumentStore(
            connection_config=OracleConnectionConfig(
                user=Secret.from_env_var("ORACLE_USER"),
                password=Secret.from_env_var("ORACLE_PASSWORD"),
                dsn=Secret.from_env_var("ORACLE_DSN"),
            ),
            embedding_dim=1536,
        )
    """

    def __init__(
        self,
        *,
        connection_config: OracleConnectionConfig,
        table_name: str = "haystack_documents",
        embedding_dim: int,
        distance_metric: Literal["COSINE", "EUCLIDEAN", "DOT"] = "COSINE",
        create_table_if_not_exists: bool = True,
        create_index: bool = False,
        hnsw_neighbors: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_accuracy: int = 95,
        hnsw_parallel: int = 4,
    ) -> None:
        if not _SAFE_TABLE_NAME.match(table_name):
            msg = (
                f"Invalid table_name {table_name!r}. Must be a valid Oracle identifier "
                "(letters, digits, _, $, # — max 128 chars, cannot start with a digit)."
            )
            raise ValueError(msg)
        if embedding_dim <= 0:
            msg = f"embedding_dim must be a positive integer, got {embedding_dim}"
            raise ValueError(msg)

        self.connection_config = connection_config
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.create_table_if_not_exists = create_table_if_not_exists
        self.create_index = create_index
        self.hnsw_neighbors = hnsw_neighbors
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_accuracy = hnsw_accuracy
        self.hnsw_parallel = hnsw_parallel

        self._pool: oracledb.ConnectionPool | None = None
        self._pool_lock = threading.Lock()

        if create_table_if_not_exists:
            self._ensure_table()
        if create_index:
            self.create_hnsw_index()

    def _get_pool(self) -> oracledb.ConnectionPool:
        if self._pool is not None:
            return self._pool
        with self._pool_lock:
            if self._pool is not None:
                return self._pool

            cfg = self.connection_config
            password = cfg.password.resolve_value()

            connect_kwargs: dict[str, Any] = {
                "user": cfg.user.resolve_value(),
                "password": password,
                "dsn": cfg.dsn.resolve_value(),
                "min": cfg.min_connections,
                "max": cfg.max_connections,
                "increment": 1,
            }
            if cfg.wallet_location:
                connect_kwargs["config_dir"] = cfg.wallet_location
                connect_kwargs["wallet_location"] = cfg.wallet_location
                connect_kwargs["wallet_password"] = (
                    cfg.wallet_password.resolve_value() if cfg.wallet_password else password
                )

            self._pool = oracledb.create_pool(**connect_kwargs)
        return self._pool

    def _get_connection(self) -> oracledb.Connection:
        return self._get_pool().acquire()

    def __del__(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                logger.warning("Failed to close Oracle connection pool during cleanup.", exc_info=True)

    def _ensure_table(self) -> None:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id        VARCHAR2(64) PRIMARY KEY,
                text      CLOB,
                metadata  JSON,
                embedding VECTOR({self.embedding_dim}, FLOAT32)
            )
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()

        self._ensure_keyword_index()

    def _ensure_keyword_index(self) -> None:
        index_name = f"{self.table_name}_search_idx"
        if len(index_name) > MAX_INDEX_NAME_LEN:
            index_name = index_name[:MAX_INDEX_NAME_LEN]
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"BEGIN DBMS_SEARCH.CREATE_INDEX('{index_name}'); "
                    f"DBMS_SEARCH.ADD_SOURCE('{index_name}', '{self.table_name}'); END;"
                )
                conn.commit()
        except oracledb.DatabaseError as e:
            logger.debug("Could not create keyword index (may already exist): %s", e)

    def create_hnsw_index(self) -> None:
        """
        Create an HNSW vector index on the embedding column.

        Safe to call multiple times — uses IF NOT EXISTS.
        """
        sql = f"""
            CREATE VECTOR INDEX IF NOT EXISTS {self.table_name}_vidx
            ON {self.table_name}(embedding)
            ORGANIZATION INMEMORY NEIGHBOR GRAPH
            WITH TARGET ACCURACY {self.hnsw_accuracy}
            DISTANCE {self.distance_metric}
            PARAMETERS (type HNSW, neighbors {self.hnsw_neighbors},
                        efConstruction {self.hnsw_ef_construction})
            PARALLEL {self.hnsw_parallel}
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()

    async def create_hnsw_index_async(self) -> None:
        """Async variant of create_hnsw_index."""
        await asyncio.to_thread(self.create_hnsw_index)

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
            and the policy is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.
        :returns: The number of documents written to the document store.
        """
        if not isinstance(documents, list):
            msg = "write_documents expects a list of Document objects."
            raise ValueError(msg)
        if documents and not isinstance(documents[0], Document):
            msg = "write_documents expects a list of Document objects."
            raise ValueError(msg)
        if not documents:
            return 0
        if policy in (DuplicatePolicy.NONE, DuplicatePolicy.FAIL):
            return self._insert_documents(documents)
        if policy == DuplicatePolicy.SKIP:
            return self._skip_duplicate_documents(documents)
        if policy == DuplicatePolicy.OVERWRITE:
            return self._upsert_documents(documents)
        msg = f"Unknown DuplicatePolicy: {policy}"
        raise ValueError(msg)

    @staticmethod
    def _to_row(doc: Document) -> tuple[str, str | None, str, _array.array | None]:
        """
        Convert a Document to (id, text, metadata_json, embedding_array).

        Haystack IDs are stored verbatim in a VARCHAR2(64) column, so any
        string ID (UUID, SHA-256 hash, or custom) is accepted without conversion.
        """
        doc_id = doc.id
        text = doc.content
        meta = json.dumps(doc.meta or {})
        emb: _array.array | None = None
        if doc.embedding is not None:
            emb = _array.array("f", doc.embedding)
        return doc_id, text, meta, emb

    @staticmethod
    def _to_named_row(doc: Document) -> dict[str, Any]:
        doc_id, text, meta, emb = OracleDocumentStore._to_row(doc)
        return {"doc_id": doc_id, "doc_text": text, "doc_meta": meta, "doc_emb": emb}

    def _insert_documents(self, documents: list[Document]) -> int:
        sql = f"""
            INSERT INTO {self.table_name} (id, text, metadata, embedding)
            VALUES (:doc_id, :doc_text, :doc_meta, :doc_emb)
        """
        rows = [OracleDocumentStore._to_named_row(d) for d in documents]
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                cur.executemany(sql, rows)
                conn.commit()
        except oracledb.IntegrityError as exc:
            msg = f"Document already exists. Use DuplicatePolicy.OVERWRITE or SKIP. Original error: {exc}"
            raise DuplicateDocumentError(msg) from exc
        return len(rows)

    def _skip_duplicate_documents(self, documents: list[Document]) -> int:
        # For a MERGE with WHEN NOT MATCHED only, Oracle reports 0 rows affected
        # for existing documents and 1 for each new insert. oracledb sums these
        # across executemany iterations, so cur.rowcount equals the number of
        # newly written documents.
        sql = f"""
            MERGE INTO {self.table_name} t
            USING (SELECT :doc_id AS id FROM dual) s ON (t.id = s.id)
            WHEN NOT MATCHED THEN
                INSERT (id, text, metadata, embedding)
                VALUES (s.id, :doc_text, :doc_meta, :doc_emb)
        """
        rows = [OracleDocumentStore._to_named_row(d) for d in documents]
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            written = cur.rowcount
            conn.commit()
        return written

    def _upsert_documents(self, documents: list[Document]) -> int:
        sql = f"""
            MERGE INTO {self.table_name} t
            USING (SELECT :doc_id AS id FROM dual) s ON (t.id = s.id)
            WHEN MATCHED THEN
                UPDATE SET t.text = :doc_text, t.metadata = :doc_meta, t.embedding = :doc_emb
            WHEN NOT MATCHED THEN
                INSERT (id, text, metadata, embedding)
                VALUES (s.id, :doc_text, :doc_meta, :doc_emb)
        """
        rows = [OracleDocumentStore._to_named_row(d) for d in documents]
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            written = cur.rowcount
            conn.commit()
        return written

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Asynchronously writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
            and the policy is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.
        :returns: The number of documents written to the document store.
        """
        return await asyncio.to_thread(self.write_documents, documents, policy)

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        if not filters:
            return "", {}
        params: dict[str, Any] = {}
        counter = [0]
        fragment = FilterTranslator().translate(filters, params, counter)
        return f"WHERE {fragment}", params

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        where, params = OracleDocumentStore._build_where(filters)
        sql = f"SELECT id, text, JSON_SERIALIZE(metadata) AS metadata FROM {self.table_name} {where}"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [OracleDocumentStore._row_to_document(r) for r in rows]

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        return await asyncio.to_thread(self.filter_documents, filters)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return
        placeholders = ", ".join(f":p{i}" for i in range(len(document_ids)))
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        params = {f"p{i}": doc_id for i, doc_id in enumerate(document_ids)}
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        await asyncio.to_thread(self.delete_documents, document_ids)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        sql = f"SELECT COUNT(*) FROM {self.table_name}"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return row[0] if row else 0

    async def count_documents_async(self) -> int:
        """
        Asynchronously returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        return await asyncio.to_thread(self.count_documents)

    def delete_table(self) -> None:
        """Deletes the document store table."""
        with self._get_connection() as conn, conn.cursor() as cur:
            try:
                cur.execute(f"DROP TABLE {self.table_name} PURGE")
            except oracledb.DatabaseError:
                pass
            index_name = f"{self.table_name}_search_idx"
            if len(index_name) > MAX_INDEX_NAME_LEN:
                index_name = index_name[:MAX_INDEX_NAME_LEN]
            try:
                cur.execute(f"BEGIN DBMS_SEARCH.DROP_INDEX('{index_name}'); END;")
            except oracledb.DatabaseError:
                pass
            conn.commit()

    async def delete_table_async(self) -> None:
        """Async variant of :meth:`delete_table`."""
        await asyncio.to_thread(self.delete_table)

    def delete_all_documents(self) -> None:
        """Deletes all documents from the document store."""
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name}")
            conn.commit()

    async def delete_all_documents_async(self) -> None:
        """Async variant of :meth:`delete_all_documents`."""
        await asyncio.to_thread(self.delete_all_documents)

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the number of documents that match the provided filters.
        """
        where, params = OracleDocumentStore._build_where(filters)
        sql = f"SELECT COUNT(*) FROM {self.table_name} {where}"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return row[0] if row else 0

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """Async variant of :meth:`count_documents_by_filter`."""
        return await asyncio.to_thread(self.count_documents_by_filter, filters)

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """Deletes documents matching the provided filters and returns the count."""
        if not filters:
            return 0
        where, params = OracleDocumentStore._build_where(filters)
        sql = f"DELETE FROM {self.table_name} {where}"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            deleted = cur.rowcount
            conn.commit()
        return deleted

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """Async variant of :meth:`delete_by_filter`."""
        return await asyncio.to_thread(self.delete_by_filter, filters)

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """Updates the metadata of documents matching the filters."""
        if not meta:
            msg = "meta must be a non-empty dictionary"
            raise ValueError(msg)
        where, params = OracleDocumentStore._build_where(filters)
        sql = f"UPDATE {self.table_name} SET metadata = JSON_MERGEPATCH(metadata, :meta_patch) {where}"
        params["meta_patch"] = json.dumps(meta)
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            updated = cur.rowcount
            conn.commit()
        return updated

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """Async variant of :meth:`update_by_filter`."""
        return await asyncio.to_thread(self.update_by_filter, filters, meta)

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """Counts the unique occurrences of metadata fields."""
        if not metadata_fields:
            msg = "metadata_fields must be a non-empty list of strings"
            raise ValueError(msg)
        where, params = OracleDocumentStore._build_where(filters)
        results = {}
        with self._get_connection() as conn, conn.cursor() as cur:
            for field in metadata_fields:
                field_path = field[5:] if field.startswith("meta.") else field
                _validate_field_path(field_path)
                sql = f"SELECT COUNT(DISTINCT JSON_VALUE(metadata, '$.{field_path}')) FROM {self.table_name} {where}"
                cur.execute(sql, params)
                row = cur.fetchone()
                results[field] = row[0] if row else 0
        return results

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """Async variant of :meth:`count_unique_metadata_by_filter`."""
        return await asyncio.to_thread(self.count_unique_metadata_by_filter, filters, metadata_fields)

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Return a mapping of metadata field names to their detected types.

        Uses Oracle's ``JSON_DATAGUIDE`` aggregate to introspect the
        stored metadata column.  Returns an empty dict when the table
        has no documents.
        """
        sql = f"SELECT JSON_DATAGUIDE(metadata) FROM {self.table_name}"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if not row or not row[0]:
                return {}
            raw_guide = row[0].read() if hasattr(row[0], "read") else row[0]
            if not raw_guide:
                return {}
            fields: dict[str, dict[str, str]] = {}
            dataguide = json.loads(raw_guide)
            for path_info in dataguide:
                path = path_info.get("o:path", "")
                if path.startswith("$."):
                    field_name = path[2:]
                    type_str = path_info.get("type", "string")
                    if type_str == "string":
                        type_str = "text"
                    fields[field_name] = {"type": type_str}
            return fields

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        Return the min and max values of a metadata field.

        Returns ``{"min": None, "max": None}`` when the table is empty
        or the field does not exist.  Numeric strings returned by
        ``JSON_VALUE`` are automatically converted to ``int`` / ``float``.

        The method first attempts numeric comparison via ``TO_NUMBER``
        so that ``MAX(1, 5, 10)`` returns ``10`` rather than ``"5"``
        (which would win under lexicographic string ordering).  If the
        field contains non-numeric values, it falls back to plain
        string comparison.
        """
        field_path = metadata_field[5:] if metadata_field.startswith("meta.") else metadata_field
        _validate_field_path(field_path)
        jv = f"JSON_VALUE(metadata, '$.{field_path}')"
        # Try numeric comparison first — correct ordering for ints/floats.
        sql_num = f"SELECT MIN(TO_NUMBER({jv})), MAX(TO_NUMBER({jv})) FROM {self.table_name} WHERE {jv} IS NOT NULL"
        with self._get_connection() as conn, conn.cursor() as cur:
            try:
                cur.execute(sql_num)
                row = cur.fetchone()
                if row and row[0] is not None:
                    return {"min": _try_parse_number(row[0]), "max": _try_parse_number(row[1])}
            except oracledb.DatabaseError:
                pass
            # Fall back to string comparison for non-numeric fields.
            sql_str = f"SELECT MIN({jv}), MAX({jv}) FROM {self.table_name}"
            cur.execute(sql_str)
            row = cur.fetchone()
            if not row or row[0] is None or row[1] is None:
                return {"min": None, "max": None}
            return {"min": _try_parse_number(row[0]), "max": _try_parse_number(row[1])}

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: str | None = None, from_: int = 0, size: int | None = None
    ) -> tuple[list[str], int]:
        """Return unique values for a specific metadata field."""
        field_path = metadata_field[5:] if metadata_field.startswith("meta.") else metadata_field
        _validate_field_path(field_path)
        base_sql = f"FROM {self.table_name} WHERE JSON_VALUE(metadata, '$.{field_path}') IS NOT NULL"
        params: dict[str, Any] = {}
        if search_term:
            base_sql += f" AND (text LIKE :search OR JSON_VALUE(metadata, '$.{field_path}') LIKE :search)"
            params["search"] = f"%{search_term}%"

        sql_count = f"SELECT COUNT(DISTINCT JSON_VALUE(metadata, '$.{field_path}')) {base_sql}"
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql_count, params)
            total = cur.fetchone()[0] or 0

            sql_vals = f"SELECT DISTINCT JSON_VALUE(metadata, '$.{field_path}') {base_sql} ORDER BY 1"
            if size is not None:
                sql_vals += " OFFSET :from_ ROWS FETCH NEXT :size ROWS ONLY"
                params["from_"] = from_
                params["size"] = size
            else:
                sql_vals += " OFFSET :from_ ROWS"
                params["from_"] = from_
            cur.execute(sql_vals, params)
            rows = cur.fetchall()
            return [str(r[0]) for r in rows], total

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """Async variant of :meth:`get_metadata_fields_info`."""
        return await asyncio.to_thread(self.get_metadata_fields_info)

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, Any]:
        """Async variant of :meth:`get_metadata_field_min_max`."""
        return await asyncio.to_thread(self.get_metadata_field_min_max, metadata_field)

    async def get_metadata_field_unique_values_async(
        self, metadata_field: str, search_term: str | None = None, from_: int = 0, size: int | None = None
    ) -> tuple[list[str], int]:
        """Async variant of :meth:`get_metadata_field_unique_values`."""
        return await asyncio.to_thread(self.get_metadata_field_unique_values, metadata_field, search_term, from_, size)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        # Oracle vector_distance() returns lower values for more similar vectors
        # across all metrics (COSINE, EUCLIDEAN, DOT) → always sort ASC.
        order = "ASC"
        where, params = OracleDocumentStore._build_where(filters)
        sql = f"""
            SELECT id, text, JSON_SERIALIZE(metadata) AS metadata,
                   vector_distance(embedding, :query_vec, {self.distance_metric}) AS score
            FROM {self.table_name}
            {where}
            ORDER BY score {order}
            FETCH APPROX FIRST :top_k ROWS ONLY
        """
        params["query_vec"] = _array.array("f", query_embedding)
        params["top_k"] = top_k
        with self._get_connection() as conn, conn.cursor() as cur:
            try:
                cur.execute(sql, params)
            except oracledb.DatabaseError as e:
                logger.debug("Embedding retrieval failed. SQL: %s\nParams: %s", sql, params)
                msg = f"Embedding retrieval failed. Error: {e!r}. You can find the SQL query and the parameters in the debug logs."
                raise DocumentStoreError(msg) from e
            rows = cur.fetchall()
        return [OracleDocumentStore._row_to_document(r, with_score=True) for r in rows]

    async def _embedding_retrieval_async(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._embedding_retrieval,
            query_embedding,
            filters=filters,
            top_k=top_k,
        )

    def _keyword_retrieval(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10
    ) -> list[Document]:
        index_name = f"{self.table_name}_search_idx"
        if len(index_name) > MAX_INDEX_NAME_LEN:
            index_name = index_name[:MAX_INDEX_NAME_LEN]
        where, params = OracleDocumentStore._build_where(filters)
        where_cond = where.replace("WHERE", "WHERE t.") if where else ""
        sql = f"""
            WITH hits AS (
                SELECT JSON_VALUE(METADATA, '$.KEY.ID') AS hit_id, SCORE(1) AS score
                FROM {index_name}
                WHERE CONTAINS(DATA, :query, 1) > 0
                ORDER BY score DESC
                FETCH APPROX FIRST :top_k ROWS ONLY
            )
            SELECT t.id, t.text, JSON_SERIALIZE(t.metadata) AS metadata, hits.score
            FROM hits
            JOIN {self.table_name} t ON t.id = hits.hit_id
            {where_cond}
            ORDER BY hits.score DESC
        """
        params["query"] = query
        params["top_k"] = top_k
        with self._get_connection() as conn, conn.cursor() as cur:
            try:
                cur.execute(sql, params)
            except oracledb.DatabaseError as e:
                logger.debug("Keyword retrieval failed. SQL: %s\nParams: %s", sql, params)
                msg = f"Keyword retrieval failed. Error: {e!r}. You can find the SQL query and the parameters in the debug logs."
                raise DocumentStoreError(msg) from e
            rows = cur.fetchall()
            return [OracleDocumentStore._row_to_document(r, with_score=True) for r in rows]

    async def _keyword_retrieval_async(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10
    ) -> list[Document]:
        return await asyncio.to_thread(self._keyword_retrieval, query, filters=filters, top_k=top_k)

    @staticmethod
    def _row_to_document(row: tuple, *, with_score: bool = False) -> Document:
        if with_score:
            raw_id, text, metadata_raw, score = row
        else:
            raw_id, text, metadata_raw, score = *row, None

        # oracledb returns CLOB/JSON as LOB objects — read them to strings
        if hasattr(text, "read"):
            text = text.read()
        if hasattr(metadata_raw, "read"):
            metadata_raw = metadata_raw.read()

        if isinstance(metadata_raw, str):
            meta = json.loads(metadata_raw)
        elif isinstance(metadata_raw, dict):
            meta = metadata_raw
        else:
            meta = {}

        return Document(
            id=raw_id,
            content=text,
            meta=meta,
            score=float(score) if score is not None else None,
            embedding=None,
            blob=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_config=self.connection_config.to_dict(),
            table_name=self.table_name,
            embedding_dim=self.embedding_dim,
            distance_metric=self.distance_metric,
            create_table_if_not_exists=self.create_table_if_not_exists,
            create_index=self.create_index,
            hnsw_neighbors=self.hnsw_neighbors,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_accuracy=self.hnsw_accuracy,
            hnsw_parallel=self.hnsw_parallel,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        params = data.get("init_parameters", {})
        if "connection_config" in params:
            params["connection_config"] = OracleConnectionConfig.from_dict(params["connection_config"])
        return default_from_dict(cls, data)
