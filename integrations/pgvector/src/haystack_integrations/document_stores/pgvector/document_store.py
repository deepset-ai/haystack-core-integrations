# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Literal, Optional

import psycopg
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Json

from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)

TABLE_DEFINITION = [
    ("id", "VARCHAR(128)", "PRIMARY KEY"),
    ("embedding", "VECTOR({embedding_dimension})"),
    ("content", "TEXT"),
    ("dataframe", "JSON"),
    ("blob_data", "BYTEA"),
    ("blob_meta", "JSON"),
    ("blob_mime_type", "VARCHAR(255)"),
    ("meta", "JSON"),
]

SIMILARITY_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_distance": "vector_cosine_ops",
    "max_inner_product": "vector_ip_ops",
    "l2_distance": "vector_l2_ops",
}

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]

HNSW_INDEX_NAME = "haystack_hnsw_index"


class PgvectorDocumentStore:
    def __init__(
        self,
        *,
        connection_string: str,
        table_name: str = "haystack_documents",
        embedding_dimension: int = 768,
        embedding_similarity_function: Literal[
            "cosine_distance", "max_inner_product", "l2_distance"
        ] = "cosine_distance",
        recreate_table: bool = False,
        search_strategy: Literal["exact_nearest_neighbor", "hnsw"] = "exact_nearest_neighbor",
        hnsw_recreate_index_if_exists: bool = False,
        hnsw_index_creation_kwargs: Optional[Dict[str, Any]] = None,
        hnsw_ef_search: Optional[int] = None,
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.embedding_similarity_function = embedding_similarity_function
        self.recreate_table = recreate_table
        self.search_strategy = search_strategy
        self.hnsw_recreate_index_if_exists = hnsw_recreate_index_if_exists
        self.hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}
        self.hnsw_ef_search = hnsw_ef_search

        connection = psycopg.connect(connection_string)
        connection.autocommit = True
        self._connection = connection
        self._cursor = connection.cursor()
        self._dict_cursor = connection.cursor(row_factory=dict_row)

        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)

        if recreate_table:
            self.delete_table()
        self._create_table_if_not_exists()

        if search_strategy == "hnsw":
            self._handle_hnsw()

    def _execute_sql(self, sql, params: Optional[tuple] = None, error_msg="", cursor=None):
        params = params or ()
        cursor = cursor or self._cursor

        try:
            result = cursor.execute(sql, params)
        except psycopg.Error as e:
            self._connection.rollback()
            raise DocumentStoreError(error_msg) from e
        return result

    def _create_table_if_not_exists(self):
        table_structure_str = ", ".join(
            f"{col[0]} {col[1]} {col[2]}" if len(col) == 3 else f"{col[0]} {col[1]}"  # noqa: PLR2004
            for col in TABLE_DEFINITION
        )

        create_sql = SQL("CREATE TABLE IF NOT EXISTS {table_name} (" + table_structure_str + ")").format(
            table_name=Identifier(self.table_name), embedding_dimension=SQLLiteral(self.embedding_dimension)
        )

        self._execute_sql(create_sql, error_msg="Could not create table in PgvectorDocumentStore")

    def delete_table(self):
        delete_sql = SQL("DROP TABLE IF EXISTS {}").format(Identifier(self.table_name))

        self._execute_sql(delete_sql, error_msg="Could not delete table in PgvectorDocumentStore")

    def _handle_hnsw(self):
        if self.hnsw_ef_search:
            sql_set_hnsw_ef_search = SQL("SET hnsw.ef_search = {hnsw_ef_search}").format(
                hnsw_ef_search=SQLLiteral(self.hnsw_ef_search)
            )
            self._execute_sql(sql_set_hnsw_ef_search, error_msg="Could not set hnsw.ef_search")

        index_esists = bool(
            self._execute_sql(
                "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s",
                (self.table_name, HNSW_INDEX_NAME),
                "Could not check if HNSW index exists",
            ).fetchone()
        )

        if index_esists and not self.hnsw_recreate_index_if_exists:
            logger.warning(
                "HNSW index already exists and won't be recreated. "
                "If you want to recreate it, set hnsw_recreate_index=True"
            )
            return

        sql_drop_index = SQL("DROP INDEX IF EXISTS {index_name}").format(index_name=Identifier(HNSW_INDEX_NAME))
        self._execute_sql(sql_drop_index, error_msg="Could not drop HNSW index")

        self._create_hnsw_index()

    def _create_hnsw_index(self):
        pg_ops = SIMILARITY_FUNCTION_TO_POSTGRESQL_OPS[self.embedding_similarity_function]
        effective_hnsw_index_creation_kwargs = {
            key: value
            for key, value in self.hnsw_index_creation_kwargs.items()
            if key in HNSW_INDEX_CREATION_VALID_KWARGS
        }

        sql_create_index = SQL("CREATE INDEX {index_name} ON {table_name} USING hnsw (embedding {ops}) ").format(
            index_name=Identifier(HNSW_INDEX_NAME), table_name=Identifier(self.table_name), ops=SQL(pg_ops)
        )

        if effective_hnsw_index_creation_kwargs:
            effective_hnsw_index_creation_kwargs_str = ", ".join(
                f"{key} = {value}" for key, value in effective_hnsw_index_creation_kwargs.items()
            )
            sql_add_creation_kwargs = SQL("WITH ({creation_kwargs_str})").format(
                creation_kwargs_str=SQL(effective_hnsw_index_creation_kwargs_str)
            )
            sql_create_index = sql_create_index + sql_add_creation_kwargs

        self._execute_sql(sql_create_index, error_msg="Could not create HNSW index")

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        sql_count = SQL("SELECT COUNT(*) FROM {}").format(Identifier(self.table_name))

        count = self._execute_sql(sql_count, error_msg="Could not count documents in PgvectorDocumentStore").fetchone()[
            0
        ]
        return count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:  # noqa: ARG002
        sql_get_docs = SQL("SELECT * FROM {table_name}").format(table_name=Identifier(self.table_name))

        self._execute_sql(
            sql_get_docs, error_msg="Could not filter documents from PgvectorDocumentStore", cursor=self._dict_cursor
        )

        # Fetch all the records
        records = self._dict_cursor.fetchall()
        docs = self._from_pg_to_haystack_documents(records)
        return docs

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents into to PgvectorDocumentStore.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to DuplicatePolicy.FAIL (or not specified).
        :return: The number of documents written to the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        db_documents = self._from_haystack_to_pg_documents(documents)

        columns_str = "(" + ", ".join(col for col, *_ in TABLE_DEFINITION) + ")"
        values_placeholder_str = "VALUES (" + ", ".join(f"%({col})s" for col, *_ in TABLE_DEFINITION) + ")"

        insert_statement = SQL("INSERT INTO {table_name} " + columns_str + " " + values_placeholder_str).format(
            table_name=Identifier(self.table_name)
        )

        if policy == DuplicatePolicy.OVERWRITE:
            update_statement = SQL(
                "ON CONFLICT (id) DO UPDATE SET "
                + ", ".join(f"{col} = EXCLUDED.{col}" for col, *_ in TABLE_DEFINITION if col != "id")
            )
            insert_statement += update_statement
        elif policy == DuplicatePolicy.SKIP:
            insert_statement += SQL("ON CONFLICT DO NOTHING")

        insert_statement += SQL(" RETURNING id")

        try:
            self._cursor.executemany(insert_statement, db_documents, returning=True)
        except psycopg.Error as e:
            self._connection.rollback()
            raise DuplicateDocumentError from e

        # get the number of the inserted documents, inspired by psycopg3 docs
        # https://www.psycopg.org/psycopg3/docs/api/cursors.html#psycopg.Cursor.executemany
        written_docs = 0
        while True:
            if self._cursor.fetchone():
                written_docs += 1
            if not self._cursor.nextset():
                break

        return written_docs

    def _from_haystack_to_pg_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        db_documents = []
        for document in documents:
            db_document = document.to_dict(flatten=False)
            db_document.pop("score")
            db_document.pop("blob")

            blob = document.blob

            blob_data, blob_meta, blob_mime_type = None, None, None

            if blob:
                blob_data = blob.data
                if blob.meta:
                    blob_meta = blob.meta
                if blob.mime_type:
                    blob_mime_type = blob.mime_type

            db_document["blob_data"] = blob_data
            db_document["blob_meta"] = Json(blob_meta)
            db_document["blob_mime_type"] = blob_mime_type

            db_document["dataframe"] = Json(document.dataframe) if document.dataframe else None
            db_document["meta"] = Json(document.meta)

            db_documents.append(db_document)

        return db_documents

    def _from_pg_to_haystack_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        haystack_documents = []
        for document in documents:
            haystack_dict = dict(document)
            blob_data = haystack_dict.pop("blob_data")
            blob_meta = haystack_dict.pop("blob_meta")
            blob_mime_type = haystack_dict.pop("blob_mime_type")

            haystack_document = Document.from_dict(haystack_dict)

            blob = None
            if blob_data:
                blob = ByteStream(data=blob_data, meta=blob_meta, mime_type=blob_mime_type)
                haystack_document.blob = blob

            haystack_documents.append(haystack_document)

        return haystack_documents

    def delete_documents(self, document_ids: List[str]) -> None:
        if not document_ids:
            return

        document_ids_str = ", ".join(f"'{document_id}'" for document_id in document_ids)

        delete_sql = SQL("DELETE FROM {table_name} WHERE id IN ({document_ids_str})").format(
            table_name=Identifier(self.table_name), document_ids_str=SQL(document_ids_str)
        )

        self._execute_sql(delete_sql, error_msg="Could not delete documents from PgvectorDocumentStore")
