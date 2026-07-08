# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    import mariadb

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from .filters import _convert_filters_to_where_clause_and_params, _validate_filters

logger = logging.getLogger(__name__)

VALID_VECTOR_FUNCTIONS = ["cosine_similarity", "l2_distance"]

VECTOR_FUNCTION_TO_SQL = {
    "cosine_similarity": "VEC_DISTANCE_COSINE",
    "l2_distance": "VEC_DISTANCE_EUCLIDEAN",
}

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS `{table_name}` (
    id VARCHAR(128) PRIMARY KEY,
    embedding VECTOR({embedding_dimension}) COMMENT 'MHNSW(M=16)',
    content LONGTEXT,
    blob_data LONGBLOB,
    blob_meta JSON,
    blob_mime_type VARCHAR(255),
    meta JSON,
    FULLTEXT KEY content_ft_idx (content)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

INSERT_STATEMENT = """
INSERT INTO `{table_name}`
(id, embedding, content, blob_data, blob_meta, blob_mime_type, meta)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_IGNORE_STATEMENT = """
INSERT IGNORE INTO `{table_name}`
(id, embedding, content, blob_data, blob_meta, blob_mime_type, meta)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

UPSERT_STATEMENT = """
INSERT INTO `{table_name}`
(id, embedding, content, blob_data, blob_meta, blob_mime_type, meta)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON DUPLICATE KEY UPDATE
    embedding = VALUES(embedding),
    content = VALUES(content),
    blob_data = VALUES(blob_data),
    blob_meta = VALUES(blob_meta),
    blob_mime_type = VALUES(blob_mime_type),
    meta = VALUES(meta)
"""

KEYWORD_QUERY = """
SELECT *, MATCH(content) AGAINST(? IN NATURAL LANGUAGE MODE) AS score
FROM `{table_name}`
{where_clause}
ORDER BY score DESC
LIMIT ?
"""

EMBEDDING_QUERY = """
SELECT *, {vec_func}(embedding, ?) AS score
FROM `{table_name}`
WHERE embedding IS NOT NULL
{extra_where}
ORDER BY score ASC
LIMIT ?
"""


class MariaDBDocumentStore(DocumentStore):
    """
    A Document Store backed by MariaDB 11.7+ using native VECTOR support.

    Uses MariaDB's `VECTOR` datatype with `MHNSW` indexing for approximate nearest-neighbour
    vector search, and `MATCH ... AGAINST` for full-text keyword search.

    Requires MariaDB 11.7 or later. Connect string parameters are passed individually;
    credentials are managed via Haystack `Secret` (defaults to `MARIADB_USER` and
    `MARIADB_PASSWORD` environment variables).

    ### Usage example

    ```python
    from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore

    store = MariaDBDocumentStore(
        host="localhost",
        port=3306,
        database="haystack",
        embedding_dimension=768,
    )
    store.write_documents(documents)
    ```
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 3306,
        database: str = "haystack",
        user: Secret | str = Secret.from_env_var("MARIADB_USER"),
        password: Secret | str = Secret.from_env_var("MARIADB_PASSWORD"),
        table_name: str = "haystack_documents",
        embedding_dimension: int = 768,
        vector_function: Literal["cosine_similarity", "l2_distance"] = "cosine_similarity",
        recreate_table: bool = False,
    ) -> None:
        """
        Initialize the MariaDBDocumentStore.

        :param host: MariaDB host. Defaults to `"localhost"`.
        :param port: MariaDB port. Defaults to `3306`.
        :param database: Database name.
        :param user: Database user. Reads `MARIADB_USER` env var by default.
        :param password: Database password. Reads `MARIADB_PASSWORD` env var by default.
        :param table_name: Table used to store documents.
        :param embedding_dimension: Dimension of embedding vectors.
        :param vector_function: Similarity function — `"cosine_similarity"` or `"l2_distance"`.
        :param recreate_table: Drop and recreate the table on init. **Deletes all data.**
        """
        self._connection: Optional[mariadb.Connection] = None
        self._cursor: Optional[mariadb.Cursor] = None
        self._table_initialized = False

        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, got '{vector_function}'"
            raise ValueError(msg)

        self.host = host
        self.port = port
        self.database = database
        self.user = Secret.from_token(user) if isinstance(user, str) else user
        self.password = Secret.from_token(password) if isinstance(password, str) else password
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.vector_function = vector_function
        self.recreate_table = recreate_table

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user.to_dict(),
            password=self.password.to_dict(),
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MariaDBDocumentStore":
        """Deserialize the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], ["user", "password"])
        return default_from_dict(cls, data)

    def _ensure_connection(self) -> None:
        """Lazily establish the DB connection and initialize the table."""
        import mariadb  # noqa: PLC0415

        if self._connection is not None and self._cursor is not None:
            try:
                self._connection.ping()
                return
            except mariadb.Error:
                self._close_connection()

        try:
            self._connection = mariadb.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user.resolve_value() or "",
                password=self.password.resolve_value() or "",
                autocommit=True,
            )
        except mariadb.Error as e:
            msg = (
                f"Failed to connect to MariaDB at {self.host}:{self.port}/{self.database}. "
                "Ensure MariaDB 11.7+ is running and credentials are correct."
            )
            raise DocumentStoreError(msg) from e

        self._cursor = self._connection.cursor(dictionary=True)

        if not self._table_initialized:
            self._initialize_table()

    def _initialize_table(self) -> None:
        import mariadb  # noqa: PLC0415

        if self.recreate_table:
            self._drop_table()

        sql = CREATE_TABLE_STATEMENT.format(
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
        )
        try:
            self._cursor.execute(sql)
            self._table_initialized = True
        except mariadb.Error as e:
            msg = f"Could not create table '{self.table_name}'"
            raise DocumentStoreError(msg) from e

    def _drop_table(self) -> None:
        import mariadb  # noqa: PLC0415

        try:
            self._cursor.execute(f"DROP TABLE IF EXISTS `{self.table_name}`")
            self._table_initialized = False
        except mariadb.Error as e:
            msg = f"Could not drop table '{self.table_name}'"
            raise DocumentStoreError(msg) from e

    def _close_connection(self) -> None:
        if self._cursor is not None:
            try:
                self._cursor.close()
            except Exception:  # noqa: BLE001
                pass
            self._cursor = None
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                pass
            self._connection = None
        self._table_initialized = False

    def close(self) -> None:
        """Close the database connection."""
        self._close_connection()

    def __del__(self) -> None:
        self._close_connection()

    def count_documents(self) -> int:
        """Return the number of documents in the store."""
        self._ensure_connection()
        self._cursor.execute(f"SELECT COUNT(*) AS cnt FROM `{self.table_name}`")
        row = self._cursor.fetchone()
        return row["cnt"] if row else 0

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Return documents matching the given Haystack filters.

        :param filters: Optional Haystack metadata filters.
        :returns: List of matching Documents.
        """
        _validate_filters(filters)
        self._ensure_connection()

        sql = f"SELECT * FROM `{self.table_name}`"
        params: list[Any] = []

        if filters:
            where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql += where_clause

        self._cursor.execute(sql, params)
        records = self._cursor.fetchall()
        return _rows_to_documents(records)

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Write documents to the store.

        :param documents: Documents to write.
        :param policy: `FAIL`, `OVERWRITE`, or `SKIP`.
        :returns: Number of documents written.
        """
        if not documents:
            return 0
        if not isinstance(documents[0], Document):
            msg = "param 'documents' must be a list of Document objects"
            raise ValueError(msg)

        self._ensure_connection()

        if policy == DuplicatePolicy.OVERWRITE:
            sql = UPSERT_STATEMENT.format(table_name=self.table_name)
        elif policy == DuplicatePolicy.SKIP:
            sql = INSERT_IGNORE_STATEMENT.format(table_name=self.table_name)
        else:
            sql = INSERT_STATEMENT.format(table_name=self.table_name)

        import mariadb  # noqa: PLC0415

        written = 0
        for doc in documents:
            row = _document_to_row(doc)
            try:
                self._cursor.execute(sql, row)
                written += 1
            except mariadb.IntegrityError as e:
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"Document with id '{doc.id}' already exists") from e

        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by ID.

        :param document_ids: IDs to delete.
        """
        import mariadb  # noqa: PLC0415

        if not document_ids:
            return
        self._ensure_connection()
        placeholders = ", ".join(["?"] * len(document_ids))
        try:
            self._cursor.execute(
                f"DELETE FROM `{self.table_name}` WHERE id IN ({placeholders})",
                tuple(document_ids),
            )
        except mariadb.Error as e:
            msg = "Failed to delete documents"
            raise DocumentStoreError(msg) from e

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        vector_function: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents by vector similarity.

        :param query_embedding: Query vector.
        :param filters: Optional Haystack filters.
        :param top_k: Maximum results.
        :param vector_function: Override the store's vector function for this query.
        :returns: List of Documents ordered by similarity (most similar first).
        """
        _validate_filters(filters)
        self._ensure_connection()

        vec_func = VECTOR_FUNCTION_TO_SQL[vector_function or self.vector_function]
        embedding_bytes = _embedding_to_bytes(query_embedding)

        extra_where = ""
        params: list[Any] = [embedding_bytes]

        if filters:
            where_clause, filter_params = _convert_filters_to_where_clause_and_params(filters, operator="AND")
            extra_where = where_clause
            params.extend(filter_params)

        params.append(top_k)

        sql = EMBEDDING_QUERY.format(
            vec_func=vec_func,
            table_name=self.table_name,
            extra_where=extra_where,
        )

        self._cursor.execute(sql, params)
        records = self._cursor.fetchall()

        docs = _rows_to_documents(records)
        # VEC_DISTANCE_* returns distance (lower = more similar); convert to a positive score
        docs = [
            replace(doc, score=float(1.0 - record["score"]) if "COSINE" in vec_func else float(-record["score"]))
            if record.get("score") is not None
            else doc
            for doc, record in zip(docs, records)
        ]
        return docs

    def _keyword_retrieval(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents by full-text keyword search.

        :param query: Search query string.
        :param filters: Optional Haystack filters.
        :param top_k: Maximum results.
        :returns: List of Documents ordered by relevance.
        """
        _validate_filters(filters)
        self._ensure_connection()

        params: list[Any] = [query]
        where_clause = ""

        if filters:
            where_clause, filter_params = _convert_filters_to_where_clause_and_params(filters)
            params = [query] + filter_params

        params.append(top_k)

        sql = KEYWORD_QUERY.format(table_name=self.table_name, where_clause=where_clause)
        self._cursor.execute(sql, params)
        records = self._cursor.fetchall()

        docs = _rows_to_documents(records)
        docs = [
            replace(doc, score=float(record.get("score") or 0.0))
            for doc, record in zip(docs, records)
        ]
        return docs


def _embedding_to_bytes(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _bytes_to_embedding(data: bytes | bytearray) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _document_to_row(doc: Document) -> tuple:
    embedding_bytes = _embedding_to_bytes(doc.embedding) if doc.embedding is not None else None
    blob_data = doc.blob.data if doc.blob else None
    blob_meta = json.dumps(doc.blob.meta) if doc.blob and doc.blob.meta else None
    blob_mime_type = doc.blob.mime_type if doc.blob else None
    meta = json.dumps(doc.meta) if doc.meta else json.dumps({})
    return (doc.id, embedding_bytes, doc.content, blob_data, blob_meta, blob_mime_type, meta)


def _rows_to_documents(records: list[dict[str, Any]]) -> list[Document]:
    docs = []
    for record in records:
        row = dict(record)
        row.pop("score", None)

        blob_data = row.pop("blob_data", None)
        blob_meta_raw = row.pop("blob_meta", None)
        blob_mime_type = row.pop("blob_mime_type", None)

        if isinstance(row.get("meta"), str):
            row["meta"] = json.loads(row["meta"])
        elif row.get("meta") is None:
            row.pop("meta", None)

        emb = row.get("embedding")
        if emb is not None:
            if hasattr(emb, "tolist"):
                row["embedding"] = emb.tolist()
            elif isinstance(emb, (bytes, bytearray)):
                row["embedding"] = _bytes_to_embedding(emb)

        doc = Document.from_dict(row)

        if blob_data:
            if isinstance(blob_meta_raw, str):
                blob_meta_raw = json.loads(blob_meta_raw)
            doc = replace(doc, blob=ByteStream(data=blob_data, meta=blob_meta_raw or {}, mime_type=blob_mime_type))

        docs.append(doc)
    return docs
