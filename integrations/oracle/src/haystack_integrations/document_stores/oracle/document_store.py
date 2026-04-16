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
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .filters import FilterTranslator

logger = logging.getLogger(__name__)

_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#]{0,127}$")


@dataclass
class OracleConnectionConfig:
    """
    Connection parameters for Oracle Database.

    Supports both thin (direct TCP) and thick (wallet / ADB-S) modes.
    Thin mode requires no Oracle Instant Client; thick mode is activated
    automatically when *wallet_location* is provided.
    """

    user: str
    password: Secret
    dsn: str
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
            "user": self.user,
            "password": self.password.to_dict(),
            "dsn": self.dsn,
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
        deserialize_secrets_inplace(data, keys=["password", "wallet_password"])
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
                user="scott",
                password=Secret.from_env_var("ORACLE_PASSWORD"),
                dsn="localhost:1521/freepdb1",
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
                "user": cfg.user,
                "password": password,
                "dsn": cfg.dsn,
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
        # MERGE rowcount in Oracle reflects rows touched, not just inserted.
        # Count before/after to return an accurate number of newly written docs.
        sql = f"""
            MERGE INTO {self.table_name} t
            USING (SELECT :doc_id AS id FROM dual) s ON (t.id = s.id)
            WHEN NOT MATCHED THEN
                INSERT (id, text, metadata, embedding)
                VALUES (s.id, :doc_text, :doc_meta, :doc_emb)
        """
        rows = [OracleDocumentStore._to_named_row(d) for d in documents]
        with self._get_connection() as conn, conn.cursor() as cur:
            count_before = conn.cursor().execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
            cur.executemany(sql, rows)
            count_after = conn.cursor().execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
            conn.commit()
        return count_after - count_before

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
        sql = f"SELECT id, text, metadata FROM {self.table_name} {where}"
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
            SELECT id, text, metadata,
                   vector_distance(embedding, :query_vec, {self.distance_metric}) AS score
            FROM {self.table_name}
            {where}
            ORDER BY score {order}
            FETCH APPROX FIRST :top_k ROWS ONLY
        """
        params["query_vec"] = _array.array("f", query_embedding)
        params["top_k"] = top_k
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
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
