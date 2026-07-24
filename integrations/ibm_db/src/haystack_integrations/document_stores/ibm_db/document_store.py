# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""IBM Db2 Document Store for Haystack."""

import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, Literal

import ibm_db_dbi  # type: ignore[import-untyped]
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .filters import FilterTranslator

logger = logging.getLogger(__name__)


def _parse_embedding(embedding: Any) -> list[float] | None:
    """
    Parse embedding from Db2 VECTOR type to Python list.

    Db2's VECTOR type may be returned as a string representation like '[1.0, 2.0, 3.0]'
    or as an iterable. This function handles both cases.

    :param embedding: Embedding value from database (string, list, or None)
    :return: List of floats or None
    """
    if embedding is None:
        return None

    # If it's already a list, return it
    if isinstance(embedding, list):
        return [float(x) for x in embedding]

    # If it's a string representation, parse it
    if isinstance(embedding, str):
        try:
            # Use json.loads to safely parse the string representation
            parsed = json.loads(embedding)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to convert to list (for other iterable types)
    try:
        return [float(x) for x in embedding]
    except (TypeError, ValueError):
        return None


def _row_to_document(row: tuple) -> Document:
    """
    Convert a database row tuple to a Haystack Document.

    :param row: Tuple containing (id, content, meta_json, embedding)
    :return: Haystack Document object
    """
    doc_id, content, meta_json, embedding = row

    # Parse metadata JSON
    meta = json.loads(meta_json) if meta_json else {}

    # Convert embedding to proper format if present
    embedding_list = _parse_embedding(embedding)

    return Document(
        id=doc_id,
        content=content,
        meta=meta,
        embedding=embedding_list,
    )


class IBMDb2DocumentStore:
    """
    IBM Db2 Document Store for Haystack using vector search capabilities.

    This document store uses IBM Db2's native vector search functionality
    to store and retrieve documents with embeddings.
    """

    def __init__(
        self,
        *,
        database: str,
        hostname: str,
        username: Secret = Secret.from_env_var("DB2_USERNAME"),
        password: Secret = Secret.from_env_var("DB2_PASSWORD"),
        port: int = 50000,
        protocol: str = "TCPIP",
        schema: str | None = None,
        use_ssl: bool = False,
        ssl_certificate: str | None = None,
        connection_options: dict[str, Any] | None = None,
        table_name: str = "haystack_documents",
        embedding_dim: int = 768,
        distance_metric: Literal["EUCLIDEAN", "COSINE", "MANHATTAN"] = "COSINE",
        recreate_table: bool = False,
    ):
        """
        Initialize the IBM Db2 Document Store.

        :param database: Database name
        :param hostname: Database server hostname
        :param username: Database username as a `Secret`, e.g. `Secret.from_env_var("DB2_USERNAME")`.
        :param password: Database password as a `Secret`, e.g. `Secret.from_env_var("DB2_PASSWORD")`.
        :param port: Database server port (default: 50000)
        :param protocol: Connection protocol (default: "TCPIP")
        :param schema: Database schema (optional)
        :param use_ssl: Enable SSL/TLS connection (default: False)
        :param ssl_certificate: Path to SSL certificate file (optional, required if use_ssl is True)
        :param connection_options: Additional connection options as dict (optional)
        :param table_name: Name of the table to store documents (default: "haystack_documents")
        :param embedding_dim: Dimension of embedding vectors (default: 768)
        :param distance_metric: Distance metric for similarity search (default: "COSINE")
        :param recreate_table: If True, drop and recreate the table (default: False)
        """
        self.database = database
        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port
        self.protocol = protocol
        self.schema = schema
        self.use_ssl = use_ssl
        self.ssl_certificate = ssl_certificate
        self.connection_options = connection_options
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.recreate_table = recreate_table

        self._connection: ibm_db_dbi.Connection | None = None
        self._connection_lock = threading.Lock()
        self._table_initialized = False

    def _get_connection(self) -> ibm_db_dbi.Connection:
        """
        Get or create a persistent database connection and ensure the table exists.

        Thread-safe lazy initialization with SSL support.

        :return: IBM Db2 connection object
        """
        if self._connection is not None and self._table_initialized:
            return self._connection

        with self._connection_lock:
            if self._connection is None:
                # Build connection string
                dsn = f"DATABASE={self.database};HOSTNAME={self.hostname};PORT={self.port};PROTOCOL={self.protocol}"

                # Add SSL configuration if enabled
                if self.use_ssl:
                    dsn += ";SECURITY=SSL"
                    if self.ssl_certificate:
                        dsn += f";SSLServerCertificate={self.ssl_certificate}"

                # Set connection options (autocommit OFF by default)
                conn_options = {ibm_db_dbi.SQL_ATTR_AUTOCOMMIT: ibm_db_dbi.SQL_AUTOCOMMIT_OFF}
                if self.connection_options:
                    conn_options.update(self.connection_options)

                # Create persistent connection
                conn = ibm_db_dbi.pconnect(
                    dsn=dsn,
                    user=self.username.resolve_value() or "",
                    password=self.password.resolve_value() or "",
                    conn_options=conn_options,
                )

                # Set schema if specified
                if self.schema:
                    with conn.cursor() as cur:
                        try:
                            cur.execute(f"SET SCHEMA {self.schema}")
                            conn.commit()
                        except Exception as e:
                            conn.rollback()
                            msg = f"Failed to set schema {self.schema}: {e}"
                            raise RuntimeError(msg) from e

                self._connection = conn

            if not self._table_initialized:
                self._ensure_table_exists(recreate=self.recreate_table)
                self._table_initialized = True

        return self._connection

    def close(self) -> None:
        """
        Release the associated synchronous resources.
        """
        with self._connection_lock:
            if self._connection is not None:
                with suppress(Exception):
                    self._connection.close()
                self._connection = None

    @contextmanager
    def _transaction(self, error_msg: str) -> Iterator[Any]:
        """
        Yield a cursor for a unit of work, committing on success.

        On any error the transaction is rolled back and the exception is re-raised
        as a ``DocumentStoreError`` prefixed with ``error_msg``.

        :param error_msg: Human-readable prefix for the wrapped error.
        """
        conn = self._get_connection()
        with conn.cursor() as cur:
            try:
                yield cur
                conn.commit()
            except Exception as e:
                conn.rollback()
                msg = f"{error_msg}: {e}"
                raise DocumentStoreError(msg) from e

    def _ensure_table_exists(self, recreate: bool = False) -> None:
        """
        Ensure the document table exists in the database.

        :param recreate: If True, drop and recreate the table
        """
        assert self._connection is not None  # noqa: S101
        conn = self._connection

        with conn.cursor() as cur:
            if recreate:
                # Drop table if exists
                try:
                    cur.execute(f"DROP TABLE {self.table_name}")
                    conn.commit()
                except Exception:
                    # Table might not exist, ignore error
                    conn.rollback()

            # Check if table already exists
            table_exists = False
            try:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                table_exists = True
            except Exception:
                # Table doesn't exist
                pass

            if not table_exists:
                # Create table with vector support
                create_sql = (
                    f"CREATE TABLE {self.table_name} ("
                    "id VARCHAR(512) NOT NULL PRIMARY KEY, "
                    "content CLOB(2M), "
                    "meta BLOB, "
                    f"embedding VECTOR({self.embedding_dim}, FLOAT32)"
                    ")"
                )

                try:
                    cur.execute(create_sql)
                    conn.commit()
                    logger.info(f"Created table {self.table_name}")
                except Exception:
                    conn.rollback()
                    # If it still fails, raise the error
                    raise

    @staticmethod
    def _validate_embedding(embedding: list[float] | None, allow_none: bool = True) -> None:
        """
        Validate embedding format and content.

        :param embedding: The embedding to validate
        :param allow_none: Whether None embeddings are allowed
        :raises ValueError: If embedding is invalid
        :raises TypeError: If embedding has wrong type or contains non-numeric values
        """
        if embedding is None:
            if not allow_none:
                msg = "Embedding cannot be None"
                raise ValueError(msg)
            return

        if not isinstance(embedding, list):
            msg = f"Embedding must be a list, got {type(embedding).__name__}"
            raise TypeError(msg)

        if len(embedding) == 0:
            msg = "Embedding cannot be empty"
            raise ValueError(msg)

        if not all(isinstance(x, (int, float)) for x in embedding):
            msg = "All embedding values must be numeric (int or float)"
            raise TypeError(msg)

    @staticmethod
    def _to_row(doc: Document) -> tuple:
        """Convert a Document to (id, content, meta_json, embedding_str)."""
        meta_json = json.dumps(doc.meta) if doc.meta else "{}"
        embedding_str = f"{doc.embedding}" if doc.embedding else None
        return (doc.id, doc.content, meta_json, embedding_str)

    def count_documents(self) -> int:
        """
        Count all documents in the store.

        :return: Number of documents
        """
        with self._transaction("Failed to count documents") as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cur.fetchone()
            return result[0] if result else 0

    def count_documents_by_filter(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count documents that match the provided filters.

        :param filters: Filters to apply. See Haystack documentation for filter syntax.
        :return: Number of documents matching the filters
        """
        if not filters:
            return self.count_documents()

        where_clause, params = self._build_where_clause(filters)

        with self._transaction("Failed to count documents by filter") as cur:
            query = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
            cur.execute(query, params)
            result = cur.fetchone()
            return result[0] if result else 0

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents to the store.

        :param documents: List of documents to write
        :param policy: Policy for handling duplicate documents
        :return: Number of documents written
        :raises ValueError: If documents is not a list of Document objects or has invalid embeddings
        :raises TypeError: If embeddings have invalid types
        :raises DuplicateDocumentError: If a document with the same id already exists and policy is FAIL or NONE
        """
        if not isinstance(documents, list):
            msg = f"Expected a list of Document objects, got {type(documents)}"
            raise ValueError(msg)

        if not documents:
            return 0

        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"Expected Document objects, got {type(doc)}"
                raise ValueError(msg)

            # Validate embeddings if present
            if doc.embedding is not None:
                try:
                    self._validate_embedding(doc.embedding, allow_none=False)
                except (ValueError, TypeError) as e:
                    msg = f"Invalid embedding for document '{doc.id}': {e}"
                    raise type(e)(msg) from e

        if policy in (DuplicatePolicy.NONE, DuplicatePolicy.FAIL):
            return self._insert_documents(documents)
        elif policy == DuplicatePolicy.SKIP:
            return self._skip_duplicate_documents(documents)
        elif policy == DuplicatePolicy.OVERWRITE:
            return self._upsert_documents(documents)
        else:
            msg = f"Unsupported duplicate policy: {policy}"
            raise ValueError(msg)

    def _insert_documents(self, documents: list[Document]) -> int:
        """Insert documents and fail on duplicates via database integrity errors."""
        rows = [self._to_row(doc) for doc in documents]
        conn = self._get_connection()
        with conn.cursor() as cur:
            sql = (
                f"INSERT INTO {self.table_name} (id, content, meta, embedding) "
                f"VALUES (?, ?, SYSTOOLS.JSON2BSON(?), "
                f"VECTOR(CAST(? AS CLOB(100000)), {self.embedding_dim}, FLOAT32))"
            )
            try:
                cur.executemany(sql, rows)
                conn.commit()
            except Exception as e:
                conn.rollback()
                error_msg = str(e).lower()
                duplicate_indicators = (
                    "duplicate",
                    "unique",
                    "sql0803n",
                    "primary key",
                    "sqlcode=-803",
                    "sqlstate=23505",
                )
                if any(indicator in error_msg for indicator in duplicate_indicators):
                    msg = f"Document already exists. Use DuplicatePolicy.OVERWRITE or SKIP. Original error: {e}"
                    raise DuplicateDocumentError(msg) from e
                raise
        return len(documents)

    def _skip_duplicate_documents(self, documents: list[Document]) -> int:
        rows = [self._to_row(doc) for doc in documents]
        inserted_count = 0
        merge_sql = (
            f"MERGE INTO {self.table_name} AS t "
            f"USING (VALUES (?, ?, SYSTOOLS.JSON2BSON(?), "
            f"VECTOR(CAST(? AS CLOB(100000)), {self.embedding_dim}, FLOAT32))) "
            f"AS s(id, content, meta, embedding) "
            "ON t.id = s.id "
            "WHEN NOT MATCHED THEN "
            "INSERT (id, content, meta, embedding) "
            "VALUES (s.id, s.content, s.meta, s.embedding)"
        )
        with self._transaction("Failed to skip duplicate documents") as cur:
            for row in rows:
                cur.execute(merge_sql, row)
                if cur.rowcount > 0:
                    inserted_count += 1
        return inserted_count

    def _upsert_documents(self, documents: list[Document]) -> int:
        rows = [self._to_row(doc) for doc in documents]
        merge_sql = (
            f"MERGE INTO {self.table_name} AS t "
            f"USING (VALUES (?, ?, SYSTOOLS.JSON2BSON(?), "
            f"VECTOR(CAST(? AS CLOB(100000)), {self.embedding_dim}, FLOAT32))) "
            f"AS s(id, content, meta, embedding) "
            "ON t.id = s.id "
            "WHEN MATCHED THEN "
            "UPDATE SET t.content = s.content, t.meta = s.meta, t.embedding = s.embedding "
            "WHEN NOT MATCHED THEN "
            "INSERT (id, content, meta, embedding) "
            "VALUES (s.id, s.content, s.meta, s.embedding)"
        )
        with self._transaction("Failed to upsert documents") as cur:
            for row in rows:
                cur.execute(merge_sql, row)
        return len(documents)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Filter documents using SQL-based metadata and field conditions.

        :param filters: Optional filter dictionary to constrain the returned documents.
        :return: List of matching documents.
        """
        sql = f"SELECT id, content, SYSTOOLS.BSON2JSON(meta) AS meta, embedding FROM {self.table_name}"
        params: list[Any] = []
        if filters:
            where_clause, params = self._build_where_clause(filters)
            sql = f"{sql} {where_clause}"
        # Add ORDER BY to ensure consistent ordering for test reproducibility
        sql = f"{sql} ORDER BY id"
        with self._transaction("Failed to filter documents") as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [_row_to_document(row) for row in rows]

    def _build_where_clause(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """
        Build WHERE clause from filter dictionary using FilterTranslator.

        :param filters: Filter dictionary
        :return: Tuple of (where_clause, parameters)
        """
        if not filters:
            return "", []

        params: list[Any] = []
        translator = FilterTranslator()
        where_expression = translator.translate(filters, params)
        return f"WHERE {where_expression}", params

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        :param document_ids: List of document IDs to delete
        """
        if not document_ids:
            return

        placeholders = ", ".join("?" for _ in document_ids)
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        with self._transaction("Failed to delete documents") as cur:
            cur.execute(sql, document_ids)

    def delete_by_filter(self, filters: dict[str, Any] | None = None) -> int:
        """
        Delete documents that match the provided filters.

        :param filters: Filters to apply. See Haystack documentation for filter syntax.
        :return: Number of documents deleted
        """
        if not filters:
            return 0

        where_clause, params = self._build_where_clause(filters)

        with self._transaction("Failed to delete documents by filter") as cur:
            cur.execute(f"DELETE FROM {self.table_name} {where_clause}", params)
            return cur.rowcount

    def delete_all_documents(self, recreate_index: bool = False) -> int:
        """
        Delete all documents from the document store.

        :param recreate_index: If True, recreate the table after deletion
        :return: Number of documents deleted
        """
        with self._transaction("Failed to delete all documents") as cur:
            # Count documents before deletion
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            deleted_count = cur.fetchone()[0]

            if recreate_index:
                # Drop and recreate the table
                self._ensure_table_exists(recreate=True)
            else:
                # Just delete all rows
                cur.execute(f"DELETE FROM {self.table_name}")

            return deleted_count

    def update_by_filter(self, filters: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> int:
        """
        Update documents that match the provided filters.

        :param filters: Filters to apply. See Haystack documentation for filter syntax.
        :param meta: Dictionary of metadata fields to update
        :return: Number of documents updated
        """
        if not meta:
            msg = "meta must be a non-empty dictionary"
            raise ValueError(msg)

        if not filters:
            return 0

        where_clause, params = self._build_where_clause(filters)

        with self._transaction("Failed to update documents by filter") as cur:
            # Db2 doesn't have a simple JSON merge operator like PostgreSQL's ||
            # We need to read, merge, and update
            # First, get the documents that match the filter
            select_sql = f"SELECT id, SYSTOOLS.BSON2JSON(meta) AS meta FROM {self.table_name} {where_clause}"
            cur.execute(select_sql, params)
            rows = cur.fetchall()

            updated_count = 0
            # Update each document
            for row in rows:
                doc_id, meta_json = row
                existing_meta = json.loads(meta_json) if meta_json else {}
                # Merge the metadata
                existing_meta.update(meta)
                merged_meta_json = json.dumps(existing_meta)

                # Update the document
                update_sql = f"UPDATE {self.table_name} SET meta = SYSTOOLS.JSON2BSON(?) WHERE id = ?"
                cur.execute(update_sql, (merged_meta_json, doc_id))
                updated_count += 1

            return updated_count

    def get_metadata_field_unique_values(self, field: str) -> list[Any]:
        """
        Get all unique values for a given metadata field.

        :param field: The metadata field name (can include 'meta.' prefix)
        :return: List of unique values for the field
        """
        # Strip 'meta.' prefix if present
        field_name = field.removeprefix("meta.")

        # Extract values from the JSON metadata field
        # Db2 has issues with DISTINCT/GROUP BY on JSON_VALUE results (SQL0134N error)
        # So we fetch all values and deduplicate in Python
        # Use RETURNING VARCHAR to explicitly specify the return type
        sql = (
            f"SELECT JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) "
            f"FROM {self.table_name} "
            f"WHERE JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) IS NOT NULL"
        )
        with self._transaction(f"Failed to get unique values for field '{field}'") as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        # Parse and deduplicate the values
        seen = set()
        values = []
        for row in rows:
            value = row[0]
            if value is not None:
                # Try to parse as JSON to get the actual type
                try:
                    parsed_value = json.loads(value)
                    # Use JSON string for deduplication to handle unhashable types
                    value_key = json.dumps(parsed_value, sort_keys=True)
                    if value_key not in seen:
                        seen.add(value_key)
                        values.append(parsed_value)
                except (json.JSONDecodeError, TypeError):
                    # If it's not valid JSON, use the string value
                    if value not in seen:
                        seen.add(value)
                        values.append(value)

        return values

    def get_metadata_field_min_max(self, field: str) -> dict[str, Any]:
        """
        Get the minimum and maximum values for a numeric metadata field.

        :param field: The metadata field name (can include 'meta.' prefix)
        :return: Dictionary with 'min' and 'max' keys
        """
        # Strip 'meta.' prefix if present
        field_name = field.removeprefix("meta.")

        conn = self._get_connection()

        with conn.cursor() as cur:
            # Try to get min/max treating the field as numeric
            # Use RETURNING VARCHAR to explicitly specify the return type for JSON_VALUE
            sql = (
                f"SELECT "
                f"MIN(CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) AS DOUBLE)), "
                f"MAX(CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) AS DOUBLE)) "
                f"FROM {self.table_name} "
                f"WHERE JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) IS NOT NULL"
            )
            try:
                cur.execute(sql)
                row = cur.fetchone()
                if row and row[0] is not None:
                    return {"min": row[0], "max": row[1]}
            except Exception:
                # If numeric cast fails, try lexicographic comparison
                sql = (
                    f"SELECT "
                    f"MIN(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000))), "
                    f"MAX(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000))) "
                    f"FROM {self.table_name} "
                    f"WHERE JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) IS NOT NULL"
                )
                try:
                    cur.execute(sql)
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        return {"min": row[0], "max": row[1]}
                except Exception as e:
                    msg = f"Failed to get min/max for field '{field}': {e}"
                    raise DocumentStoreError(msg) from e

        return {"min": None, "max": None}

    def get_metadata_fields_info(self) -> dict[str, dict[str, Any]]:
        """
        Get information about all metadata fields including their types.

        :return: Dictionary mapping field names to their type information
        """
        # Get all metadata from documents
        sql = f"SELECT SYSTOOLS.BSON2JSON(meta) AS meta FROM {self.table_name} WHERE meta IS NOT NULL"
        with self._transaction("Failed to get metadata fields info") as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        # Analyze the metadata to infer field types
        fields_info: dict[str, dict[str, Any]] = {}

        for row in rows:
            meta_json = row[0]
            if not meta_json:
                continue

            try:
                meta = json.loads(meta_json)
                if not isinstance(meta, dict):
                    continue

                for field_name, field_value in meta.items():
                    if field_name not in fields_info:
                        if field_value is not None:
                            field_type = self._infer_field_type(field_value)
                            fields_info[field_name] = {"type": field_type}
                        else:
                            fields_info[field_name] = {"type": "text"}
            except (json.JSONDecodeError, TypeError):
                continue

        return fields_info

    def count_unique_metadata_by_filter(
        self, filters: dict[str, Any] | None = None, metadata_fields: list[str] | None = None
    ) -> dict[str, int]:
        """
        Count unique values for specified metadata fields, optionally filtered.

        :param filters: Optional filters to apply before counting
        :param metadata_fields: List of metadata field names to count unique values for
        :return: Dictionary mapping field names to their unique value counts
        """
        if not metadata_fields:
            return {}

        where_clause, params = self._build_where_clause(filters) if filters else ("", [])

        result = {}
        for field in metadata_fields:
            # Strip 'meta.' prefix if present
            field_name = field.removeprefix("meta.")

            # Count distinct values for this field
            # We need to fetch all values and deduplicate in Python due to Db2's JSON handling
            sql = (
                f"SELECT JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) "
                f"FROM {self.table_name} "
            )
            if where_clause:
                sql += where_clause + " AND "
            else:
                sql += "WHERE "
            sql += f"JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_name}' RETURNING VARCHAR(1000)) IS NOT NULL"

            with self._transaction(f"Failed to count unique metadata for field '{field}'") as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            # Deduplicate values
            unique_values = set()
            for row in rows:
                value = row[0]
                if value is not None:
                    # Try to parse as JSON to handle different types consistently
                    try:
                        parsed_value = json.loads(value)
                        # Use JSON string for deduplication to handle unhashable types
                        value_key = json.dumps(parsed_value, sort_keys=True)
                        unique_values.add(value_key)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not valid JSON, use the string value
                        unique_values.add(value)

            result[field] = len(unique_values)

        return result

    @staticmethod
    def _infer_field_type(value: Any) -> str:
        """
        Infer the type of a metadata field value.

        :param value: The field value
        :return: Type string ('integer', 'real', 'boolean', or 'text')
        """
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "real"
        if isinstance(value, str):
            return "text"
        return "text"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the document store to a dictionary.

        :return: Dictionary representation
        """
        return default_to_dict(
            self,
            database=self.database,
            hostname=self.hostname,
            port=self.port,
            username=self.username.to_dict(),
            password=self.password.to_dict(),
            protocol=self.protocol,
            schema=self.schema,
            use_ssl=self.use_ssl,
            ssl_certificate=self.ssl_certificate,
            connection_options=self.connection_options,
            table_name=self.table_name,
            embedding_dim=self.embedding_dim,
            distance_metric=self.distance_metric,
            recreate_table=self.recreate_table,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IBMDb2DocumentStore":
        """
        Deserialize the document store from a dictionary.

        :param data: Dictionary representation
        :return: IBMDb2DocumentStore instance
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["username", "password"])
        return default_from_dict(cls, data)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents by embedding similarity.

        :param query_embedding: Query embedding vector
        :param filters: Optional filters to apply
        :param top_k: Number of documents to retrieve
        :return: List of documents with similarity scores
        :raises ValueError: If query_embedding is invalid
        :raises TypeError: If query_embedding has invalid type
        """
        # Validate query embedding
        self._validate_embedding(query_embedding, allow_none=False)

        conn = self._get_connection()
        embedding_str = f"{query_embedding}"
        where_clause, filter_params = self._build_where_clause(filters) if filters else ("", [])

        # Add condition to exclude NULL embeddings
        null_check = "embedding IS NOT NULL"
        if where_clause:
            where_clause = f"{where_clause} AND {null_check}"
        else:
            where_clause = f"WHERE {null_check}"

        sql = (
            f"SELECT id, content, SYSTOOLS.BSON2JSON(meta) AS meta, embedding, "
            f"VECTOR_DISTANCE(embedding, VECTOR(CAST(? AS CLOB(100000)), {self.embedding_dim}, FLOAT32), "
            f"{self.distance_metric}) AS score "
            f"FROM {self.table_name} "
            f"{where_clause} "
            f"ORDER BY score ASC FETCH FIRST ? ROWS ONLY"
        )
        params: list[Any] = [embedding_str, *filter_params, top_k]

        with conn.cursor() as cur:
            cur.execute(sql, params)
            try:
                rows = cur.fetchall()
            except BaseException as e:
                # If we get a division by zero error (SQL0801N), it means there are zero vectors
                # In this case, we'll return empty results or filter them out
                error_msg = str(e)
                # Check both the error message and the __cause__ attribute
                cause_msg = str(e.__cause__) if hasattr(e, "__cause__") and e.__cause__ else ""
                if (
                    "SQL0801N" in error_msg
                    or "Division by zero" in error_msg
                    or "SQL0801N" in cause_msg
                    or "Division by zero" in cause_msg
                ):
                    # For COSINE metric with zero vectors, return empty results
                    # This is an edge case that shouldn't happen in production
                    rows = []
                else:
                    raise

        documents = []
        for row in rows:
            doc_id, content, meta_json, embedding, score = row
            meta = json.loads(meta_json) if meta_json else {}
            embedding_list = _parse_embedding(embedding)
            doc = Document(
                id=doc_id,
                content=content,
                meta=meta,
                embedding=embedding_list,
                score=float(score),
            )
            documents.append(doc)

        return documents


__all__ = ["IBMDb2DocumentStore"]
