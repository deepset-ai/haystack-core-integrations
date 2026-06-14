# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""IBM DB2 Document Store for Haystack."""

import asyncio
import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Literal

import ibm_db_dbi

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

logger = logging.getLogger(__name__)


@dataclass
class Db2ConnectionConfig:
    """
    Configuration for IBM DB2 database connection.
    
    :param database: Database name
    :param hostname: Database server hostname
    :param port: Database server port (default: 50000)
    :param username: Database username
    :param password: Database password
    :param protocol: Connection protocol (default: "TCPIP")
    :param schema: Database schema (optional)
    :param connection_options: Additional connection options as dict (optional)
    """
    
    database: str
    hostname: str
    port: int = 50000
    username: str = ""
    password: str = ""
    protocol: str = "TCPIP"
    schema: str | None = None
    connection_options: dict[str, Any] | None = None


def _row_to_document(row: tuple) -> Document:
    """
    Convert a database row tuple to a Haystack Document.
    
    :param row: Tuple containing (id, content, meta_json, embedding)
    :return: Haystack Document object
    """
    doc_id, content, meta_json, embedding = row
    
    # Parse metadata JSON
    meta = json.loads(meta_json) if meta_json else {}
    
    # Convert embedding list to proper format if present
    embedding_list = list(embedding) if embedding else None
    
    return Document(
        id=doc_id,
        content=content,
        meta=meta,
        embedding=embedding_list,
    )


class Db2DocumentStore:
    """
    IBM DB2 Document Store for Haystack using vector search capabilities.
    
    This document store uses IBM DB2's native vector search functionality
    to store and retrieve documents with embeddings.
    """
    
    def __init__(
        self,
        *,
        connection_config: Db2ConnectionConfig,
        table_name: str = "haystack_documents",
        embedding_dim: int = 768,
        distance_metric: Literal["EUCLIDEAN", "COSINE", "MANHATTAN"] = "COSINE",
        recreate_table: bool = False,
    ):
        """
        Initialize the IBM DB2 Document Store.
        
        :param connection_config: Database connection configuration
        :param table_name: Name of the table to store documents (default: "haystack_documents")
        :param embedding_dim: Dimension of embedding vectors (default: 768)
        :param distance_metric: Distance metric for similarity search (default: "COSINE")
        :param recreate_table: If True, drop and recreate the table (default: False)
        """
        self.connection_config = connection_config
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        # Connection pool (lazy initialization)
        self._connection: ibm_db_dbi.Connection | None = None
        self._connection_lock = threading.Lock()
        
        # Initialize table
        self._ensure_table_exists(recreate=recreate_table)
    
    def _get_connection(self) -> ibm_db_dbi.Connection:
        """
        Get or create a persistent database connection.
        Thread-safe lazy initialization.
        
        :return: IBM DB2 connection object
        """
        if self._connection is None:
            with self._connection_lock:
                if self._connection is None:
                    # Build connection string
                    dsn = (
                        f"DATABASE={self.connection_config.database};"
                        f"HOSTNAME={self.connection_config.hostname};"
                        f"PORT={self.connection_config.port};"
                        f"PROTOCOL={self.connection_config.protocol}"
                    )
                    
                    # Set connection options (autocommit OFF by default)
                    conn_options = {
                        ibm_db_dbi.SQL_ATTR_AUTOCOMMIT: ibm_db_dbi.SQL_AUTOCOMMIT_OFF
                    }
                    if self.connection_config.connection_options:
                        conn_options.update(self.connection_config.connection_options)
                    
                    # Create persistent connection
                    self._connection = ibm_db_dbi.pconnect(
                        dsn=dsn,
                        user=self.connection_config.username,
                        password=self.connection_config.password,
                        conn_options=conn_options,
                    )
                    
                    # Set schema if specified
                    if self.connection_config.schema:
                        self._connection.set_current_schema(self.connection_config.schema)
        
        return self._connection
    
    def _ensure_table_exists(self, recreate: bool = False) -> None:
        """
        Ensure the document table exists in the database.
        
        :param recreate: If True, drop and recreate the table
        """
        conn = self._get_connection()
        
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
                except Exception as e:
                    conn.rollback()
                    # If it still fails, raise the error
                    raise
    
    def _to_row(self, doc: Document) -> tuple:
        """
        Convert a Haystack Document to a database row tuple.
        
        :param doc: Haystack Document object
        :return: Tuple of (id, content, meta_json, embedding_str)
        """
        # Serialize metadata to JSON
        meta_json = json.dumps(doc.meta) if doc.meta else "{}"
        
        # Convert embedding to string representation for VECTOR() function
        # LangChain uses f"{embedding}" which creates a string like "[0.1, 0.2, ...]"
        embedding_str = f"{doc.embedding}" if doc.embedding else None
        
        return (doc.id, doc.content, meta_json, embedding_str)
    
    def count_documents(self) -> int:
        """
        Count all documents in the store.
        
        :return: Number of documents
        """
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cur.fetchone()
            return result[0] if result else 0
    
    async def count_documents_async(self) -> int:
        """
        Count all documents asynchronously.
        
        :return: Number of documents
        """
        return await asyncio.to_thread(self.count_documents)
    
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
        """
        if not documents:
            return 0
        
        conn = self._get_connection()
        
        if policy == DuplicatePolicy.NONE:
            return self._insert_documents(conn, documents)
        elif policy == DuplicatePolicy.FAIL:
            return self._insert_documents_fail(conn, documents)
        elif policy == DuplicatePolicy.SKIP:
            return self._skip_duplicate_documents(conn, documents)
        elif policy == DuplicatePolicy.OVERWRITE:
            return self._upsert_documents(conn, documents)
        else:
            msg = f"Unsupported duplicate policy: {policy}"
            raise ValueError(msg)
    
    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents asynchronously.
        
        :param documents: List of documents to write
        :param policy: Policy for handling duplicate documents
        :return: Number of documents written
        """
        return await asyncio.to_thread(self.write_documents, documents, policy)
    
    def _insert_documents(self, conn: ibm_db_dbi.Connection, documents: list[Document]) -> int:
        """
        Insert documents without duplicate checking.
        
        :param conn: Database connection
        :param documents: List of documents to insert
        :return: Number of documents inserted
        """
        rows = [self._to_row(doc) for doc in documents]
        
        with conn.cursor() as cur:
            # Use executemany for batch insert
            # Convert string embedding to VECTOR using CAST to CLOB first
            sql = (
                f"INSERT INTO {self.table_name} (id, content, meta, embedding) "
                f"VALUES (?, ?, SYSTOOLS.JSON2BSON(?), "
                f"VECTOR(CAST(? AS CLOB(100000)), {self.embedding_dim}, FLOAT32))"
            )
            cur.executemany(sql, rows)
            conn.commit()
            return len(documents)
    
    def _insert_documents_fail(self, conn: ibm_db_dbi.Connection, documents: list[Document]) -> int:
        """
        Insert documents and fail on duplicates.
        
        :param conn: Database connection
        :param documents: List of documents to insert
        :return: Number of documents inserted
        :raises DuplicateDocumentError: If duplicate document IDs are found
        """
        try:
            return self._insert_documents(conn, documents)
        except Exception as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "duplicate" in error_msg or "unique" in error_msg:
                msg = f"Duplicate document IDs found: {error_msg}"
                raise DuplicateDocumentError(msg) from e
            raise
    
    def _skip_duplicate_documents(self, conn: ibm_db_dbi.Connection, documents: list[Document]) -> int:
        """
        Insert documents, skipping duplicates.
        
        :param conn: Database connection
        :param documents: List of documents to insert
        :return: Number of documents inserted
        """
        rows = [self._to_row(doc) for doc in documents]
        inserted_count = 0
        
        with conn.cursor() as cur:
            # Use MERGE with WHEN NOT MATCHED to skip duplicates
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
            
            for row in rows:
                cur.execute(merge_sql, row)
                # Check if row was inserted (rowcount > 0)
                if cur.rowcount > 0:
                    inserted_count += 1
            
            conn.commit()
        
        return inserted_count
    
    def _upsert_documents(self, conn: ibm_db_dbi.Connection, documents: list[Document]) -> int:
        """
        Insert or update documents (upsert).
        
        :param conn: Database connection
        :param documents: List of documents to upsert
        :return: Number of documents affected
        """
        rows = [self._to_row(doc) for doc in documents]
        
        with conn.cursor() as cur:
            # Use MERGE with both MATCHED AND NOT MATCHED clauses
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
            
            for row in rows:
                cur.execute(merge_sql, row)
            
            conn.commit()
        
        return len(documents)
    
    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Filter documents based on metadata using pure SQL approach.
        
        All filtering is now performed in the database using SQL WHERE clauses,
        similar to PgVector's implementation. This is more efficient and scalable.
        
        :param filters: Filter dictionary (optional)
        :return: List of matching documents
        """
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            # Build SELECT query with WHERE clause
            sql = f"SELECT id, content, SYSTOOLS.BSON2JSON(meta) AS meta, embedding FROM {self.table_name}"
            params = []
            
            if filters:
                where_clause, params = self._build_where_clause(filters)
                sql = f"{sql} {where_clause}"
            
            cur.execute(sql, params)
            rows = cur.fetchall()
            documents = [_row_to_document(row) for row in rows]
            
            return documents
    
    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Filter documents asynchronously using pure SQL approach.
        
        :param filters: Filter dictionary (optional)
        :return: List of matching documents
        """
        return await asyncio.to_thread(self.filter_documents, filters)
    
    def _build_where_clause(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """
        Build WHERE clause from filter dictionary using pure SQL.
        
        Handles both direct columns (id, content, embedding) and metadata fields
        using DB2's JSON functions. All filtering is performed in the database.
        
        :param filters: Filter dictionary
        :return: Tuple of (where_clause, parameters)
        """
        if not filters:
            return "", []
        
        params: list[Any] = []
        
        def translate(filter_dict: dict[str, Any]) -> str:
            operator = filter_dict.get("operator")
            if operator is None:
                msg = "Each filter condition must include an 'operator' key."
                raise ValueError(msg)
            
            # Handle logical operators
            if operator in ("AND", "OR", "NOT", "$and", "$or", "$not"):
                conditions = filter_dict.get("conditions", [])
                if not conditions:
                    msg = f"Logical operator {operator!r} requires a non-empty 'conditions' list."
                    raise ValueError(msg)
                
                if operator in ("NOT", "$not"):
                    if len(conditions) != 1:
                        msg = "NOT operator requires exactly one condition."
                        raise ValueError(msg)
                    return f"(NOT {translate(conditions[0])})"
                
                logical_op = "AND" if operator in ("AND", "$and") else "OR"
                translated = [translate(cond) for cond in conditions]
                return f"({f' {logical_op} '.join(translated)})"
            
            # Handle comparison operators
            field = filter_dict.get("field")
            if not field:
                msg = "Comparison filters must include a 'field' key."
                raise ValueError(msg)
            
            # Determine if this is a direct column or metadata field
            if field in ("id", "content", "embedding"):
                # Direct column access
                field_expr = field
            elif field.startswith("meta."):
                # Metadata field - extract from BSON with proper casting
                field_path = field[5:]  # Remove "meta." prefix
                # Cast to VARCHAR to ensure string comparison works correctly
                field_expr = f"CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_path}') AS VARCHAR(1000))"
            else:
                # Assume it's a metadata field without "meta." prefix
                field_expr = f"CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field}') AS VARCHAR(1000))"
            
            value = filter_dict.get("value")
            
            # Handle membership operators
            if operator in ("in", "not in", "$in", "$nin"):
                if not isinstance(value, list) or not value:
                    msg = f"Operator {operator!r} requires a non-empty list value."
                    raise ValueError(msg)
                
                placeholders = ", ".join("?" for _ in value)
                params.extend(value)
                
                if operator in ("in", "$in"):
                    return f"{field_expr} IN ({placeholders})"
                else:
                    return f"({field_expr} IS NULL OR {field_expr} NOT IN ({placeholders}))"
            
            # Handle comparison operators
            comparison_map = {
                "==": "=", "$eq": "=",
                "!=": "!=", "$ne": "!=",
                ">": ">", "$gt": ">",
                ">=": ">=", "$gte": ">=",
                "<": "<", "$lt": "<",
                "<=": "<=", "$lte": "<=",
            }
            
            sql_operator = comparison_map.get(operator)
            if sql_operator is None:
                msg = f"Unsupported filter operator: {operator!r}"
                raise ValueError(msg)
            
            params.append(value)
            return f"{field_expr} {sql_operator} ?"
        
        where_expression = translate(filters)
        return f"WHERE {where_expression}", params
    
    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs.
        
        :param document_ids: List of document IDs to delete
        """
        if not document_ids:
            return
        
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            placeholders = ", ".join("?" for _ in document_ids)
            sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
            cur.execute(sql, document_ids)
            conn.commit()
    
    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Delete documents asynchronously.
        
        :param document_ids: List of document IDs to delete
        """
        await asyncio.to_thread(self.delete_documents, document_ids)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the document store to a dictionary.
        
        :return: Dictionary representation
        """
        return default_to_dict(
            self,
            connection_config=asdict(self.connection_config),
            table_name=self.table_name,
            embedding_dim=self.embedding_dim,
            distance_metric=self.distance_metric,
        )
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Db2DocumentStore":
        """
        Deserialize the document store from a dictionary.
        
        :param data: Dictionary representation
        :return: Db2DocumentStore instance
        """
        init_params = data.get("init_parameters", {})
        
        # Reconstruct connection config
        connection_config_dict = init_params.pop("connection_config", {})
        connection_config = Db2ConnectionConfig(**connection_config_dict)
        
        # Pass connection_config separately and let default_from_dict handle other params
        return cls(connection_config=connection_config, **init_params)


__all__ = ["Db2DocumentStore", "Db2ConnectionConfig"]

# Made with Bob
