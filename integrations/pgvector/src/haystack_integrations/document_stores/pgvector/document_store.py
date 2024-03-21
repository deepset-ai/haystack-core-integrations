# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Literal, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.filters import convert
from psycopg import Error, IntegrityError, connect
from psycopg.abc import Query
from psycopg.cursor import Cursor
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Jsonb

from pgvector.psycopg import register_vector

from .filters import _convert_filters_to_where_clause_and_params

logger = logging.getLogger(__name__)

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS {table_name} (
id VARCHAR(128) PRIMARY KEY,
embedding VECTOR({embedding_dimension}),
content TEXT,
dataframe JSONB,
blob_data BYTEA,
blob_meta JSONB,
blob_mime_type VARCHAR(255),
meta JSONB)
"""

INSERT_STATEMENT = """
INSERT INTO {table_name}
(id, embedding, content, dataframe, blob_data, blob_meta, blob_mime_type, meta)
VALUES (%(id)s, %(embedding)s, %(content)s, %(dataframe)s, %(blob_data)s, %(blob_meta)s, %(blob_mime_type)s, %(meta)s)
"""

UPDATE_STATEMENT = """
ON CONFLICT (id) DO UPDATE SET
embedding = EXCLUDED.embedding,
content = EXCLUDED.content,
dataframe = EXCLUDED.dataframe,
blob_data = EXCLUDED.blob_data,
blob_meta = EXCLUDED.blob_meta,
blob_mime_type = EXCLUDED.blob_mime_type,
meta = EXCLUDED.meta
"""

VALID_VECTOR_FUNCTIONS = ["cosine_similarity", "inner_product", "l2_distance"]

VECTOR_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_similarity": "vector_cosine_ops",
    "inner_product": "vector_ip_ops",
    "l2_distance": "vector_l2_ops",
}

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]

HNSW_INDEX_NAME = "haystack_hnsw_index"


class PgvectorDocumentStore:
    """
    A Document Store using PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector) installed.
    """

    def __init__(
        self,
        *,
        connection_string: Secret = Secret.from_env_var("PG_CONN_STR"),
        table_name: str = "haystack_documents",
        embedding_dimension: int = 768,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] = "cosine_similarity",
        recreate_table: bool = False,
        search_strategy: Literal["exact_nearest_neighbor", "hnsw"] = "exact_nearest_neighbor",
        hnsw_recreate_index_if_exists: bool = False,
        hnsw_index_creation_kwargs: Optional[Dict[str, int]] = None,
        hnsw_ef_search: Optional[int] = None,
    ):
        """
        Creates a new PgvectorDocumentStore instance.
        It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
        A specific table to store Haystack documents will be created if it doesn't exist yet.

        :param connection_string: The connection string to use to connect to the PostgreSQL database, defined as an
            environment variable, e.g.: `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"`
        :param table_name: The name of the table to use to store Haystack documents.
        :param embedding_dimension: The dimension of the embedding.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            `"cosine_similarity"` and `"inner_product"` are similarity functions and
            higher scores indicate greater similarity between the documents.
            `"l2_distance"` returns the straight-line distance between vectors,
            and the most similar documents are the ones with the smallest score.
            **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
            `vector_function` passed here. Make sure subsequent queries will keep using the same
            vector similarity function in order to take advantage of the index.
        :param recreate_table: Whether to recreate the table if it already exists.
        :param search_strategy: The search strategy to use when searching for similar embeddings.
            `"exact_nearest_neighbor"` provides perfect recall but can be slow for large numbers of documents.
            `"hnsw"` is an approximate nearest neighbor search strategy,
            which trades off some accuracy for speed; it is recommended for large numbers of documents.
            **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
            `vector_function` passed here. Make sure subsequent queries will keep using the same
            vector similarity function in order to take advantage of the index.
        :param hnsw_recreate_index_if_exists: Whether to recreate the HNSW index if it already exists.
            Only used if search_strategy is set to `"hnsw"`.
        :param hnsw_index_creation_kwargs: Additional keyword arguments to pass to the HNSW index creation.
            Only used if search_strategy is set to `"hnsw"`. You can find the list of valid arguments in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw)
        :param hnsw_ef_search: The `ef_search` parameter to use at query time. Only used if search_strategy is set to
            `"hnsw"`. You can find more information about this parameter in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw)
        """

        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)
        self.vector_function = vector_function
        self.recreate_table = recreate_table
        self.search_strategy = search_strategy
        self.hnsw_recreate_index_if_exists = hnsw_recreate_index_if_exists
        self.hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}
        self.hnsw_ef_search = hnsw_ef_search

        connection = connect(self.connection_string.resolve_value())
        connection.autocommit = True
        self._connection = connection

        # we create a generic cursor and another one that returns dictionaries
        self._cursor = connection.cursor()
        self._dict_cursor = connection.cursor(row_factory=dict_row)

        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)

        if recreate_table:
            self.delete_table()
        self._create_table_if_not_exists()

        if search_strategy == "hnsw":
            self._handle_hnsw()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_string=self.connection_string.to_dict(),
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
            search_strategy=self.search_strategy,
            hnsw_recreate_index_if_exists=self.hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=self.hnsw_index_creation_kwargs,
            hnsw_ef_search=self.hnsw_ef_search,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PgvectorDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["connection_string"])
        return default_from_dict(cls, data)

    def _execute_sql(
        self, sql_query: Query, params: Optional[tuple] = None, error_msg: str = "", cursor: Optional[Cursor] = None
    ):
        """
        Internal method to execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        :param cursor: The cursor to use to execute the SQL query. Defaults to self._cursor.
        """

        params = params or ()
        cursor = cursor or self._cursor

        sql_query_str = sql_query.as_string(cursor) if not isinstance(sql_query, str) else sql_query
        logger.debug("SQL query: %s\nParameters: %s", sql_query_str, params)

        try:
            result = cursor.execute(sql_query, params)
        except Error as e:
            self._connection.rollback()
            detailed_error_msg = f"{error_msg}.\nYou can find the SQL query and the parameters in the debug logs."
            raise DocumentStoreError(detailed_error_msg) from e

        return result

    def _create_table_if_not_exists(self):
        """
        Creates the table to store Haystack documents if it doesn't exist yet.
        """

        create_sql = SQL(CREATE_TABLE_STATEMENT).format(
            table_name=Identifier(self.table_name), embedding_dimension=SQLLiteral(self.embedding_dimension)
        )

        self._execute_sql(create_sql, error_msg="Could not create table in PgvectorDocumentStore")

    def delete_table(self):
        """
        Deletes the table used to store Haystack documents.
        The name of the table (`table_name`) is defined when initializing the `PgvectorDocumentStore`.
        """

        delete_sql = SQL("DROP TABLE IF EXISTS {table_name}").format(table_name=Identifier(self.table_name))

        self._execute_sql(delete_sql, error_msg=f"Could not delete table {self.table_name} in PgvectorDocumentStore")

    def _handle_hnsw(self):
        """
        Internal method to handle the HNSW index creation.
        It also sets the `hnsw.ef_search` parameter for queries if it is specified.
        """

        if self.hnsw_ef_search:
            sql_set_hnsw_ef_search = SQL("SET hnsw.ef_search = {hnsw_ef_search}").format(
                hnsw_ef_search=SQLLiteral(self.hnsw_ef_search)
            )
            self._execute_sql(sql_set_hnsw_ef_search, error_msg="Could not set hnsw.ef_search")

        index_exists = bool(
            self._execute_sql(
                "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s",
                (self.table_name, HNSW_INDEX_NAME),
                "Could not check if HNSW index exists",
            ).fetchone()
        )

        if index_exists and not self.hnsw_recreate_index_if_exists:
            logger.warning(
                "HNSW index already exists and won't be recreated. "
                "If you want to recreate it, pass 'hnsw_recreate_index_if_exists=True' to the "
                "Document Store constructor"
            )
            return

        sql_drop_index = SQL("DROP INDEX IF EXISTS {index_name}").format(index_name=Identifier(HNSW_INDEX_NAME))
        self._execute_sql(sql_drop_index, error_msg="Could not drop HNSW index")

        self._create_hnsw_index()

    def _create_hnsw_index(self):
        """
        Internal method to create the HNSW index.
        """

        pg_ops = VECTOR_FUNCTION_TO_POSTGRESQL_OPS[self.vector_function]
        actual_hnsw_index_creation_kwargs = {
            key: value
            for key, value in self.hnsw_index_creation_kwargs.items()
            if key in HNSW_INDEX_CREATION_VALID_KWARGS
        }

        sql_create_index = SQL("CREATE INDEX {index_name} ON {table_name} USING hnsw (embedding {ops}) ").format(
            index_name=Identifier(HNSW_INDEX_NAME), table_name=Identifier(self.table_name), ops=SQL(pg_ops)
        )

        if actual_hnsw_index_creation_kwargs:
            actual_hnsw_index_creation_kwargs_str = ", ".join(
                f"{key} = {value}" for key, value in actual_hnsw_index_creation_kwargs.items()
            )
            sql_add_creation_kwargs = SQL("WITH ({creation_kwargs_str})").format(
                creation_kwargs_str=SQL(actual_hnsw_index_creation_kwargs_str)
            )
            sql_create_index += sql_add_creation_kwargs

        self._execute_sql(sql_create_index, error_msg="Could not create HNSW index")

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """

        sql_count = SQL("SELECT COUNT(*) FROM {table_name}").format(table_name=Identifier(self.table_name))

        count = self._execute_sql(sql_count, error_msg="Could not count documents in PgvectorDocumentStore").fetchone()[
            0
        ]
        return count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :raises TypeError: If `filters` is not a dictionary.
        :returns: A list of Documents that match the given filters.
        """
        if filters:
            if not isinstance(filters, dict):
                msg = "Filters must be a dictionary"
                raise TypeError(msg)
            if "operator" not in filters and "conditions" not in filters:
                filters = convert(filters)

        sql_filter = SQL("SELECT * FROM {table_name}").format(table_name=Identifier(self.table_name))

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_filter += sql_where_clause

        result = self._execute_sql(
            sql_filter,
            params,
            error_msg="Could not filter documents from PgvectorDocumentStore.",
            cursor=self._dict_cursor,
        )

        records = result.fetchall()
        docs = self._from_pg_to_haystack_documents(records)
        return docs

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :returns: The number of documents written to the document store.
        """

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        db_documents = self._from_haystack_to_pg_documents(documents)

        sql_insert = SQL(INSERT_STATEMENT).format(table_name=Identifier(self.table_name))

        if policy == DuplicatePolicy.OVERWRITE:
            sql_insert += SQL(UPDATE_STATEMENT)
        elif policy == DuplicatePolicy.SKIP:
            sql_insert += SQL("ON CONFLICT DO NOTHING")

        sql_insert += SQL(" RETURNING id")

        sql_query_str = sql_insert.as_string(self._cursor) if not isinstance(sql_insert, str) else sql_insert
        logger.debug("SQL query: %s\nParameters: %s", sql_query_str, db_documents)

        try:
            self._cursor.executemany(sql_insert, db_documents, returning=True)
        except IntegrityError as ie:
            self._connection.rollback()
            raise DuplicateDocumentError from ie
        except Error as e:
            self._connection.rollback()
            error_msg = (
                "Could not write documents to PgvectorDocumentStore. \n"
                "You can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(error_msg) from e

        # get the number of the inserted documents, inspired by psycopg3 docs
        # https://www.psycopg.org/psycopg3/docs/api/cursors.html#psycopg.Cursor.executemany
        written_docs = 0
        while True:
            if self._cursor.fetchone():
                written_docs += 1
            if not self._cursor.nextset():
                break

        return written_docs

    @staticmethod
    def _from_haystack_to_pg_documents(documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Internal method to convert a list of Haystack Documents to a list of dictionaries that can be used to insert
        documents into the PgvectorDocumentStore.
        """

        db_documents = []
        for document in documents:
            db_document = {k: v for k, v in document.to_dict(flatten=False).items() if k not in ["score", "blob"]}

            blob = document.blob
            db_document["blob_data"] = blob.data if blob else None
            db_document["blob_meta"] = Jsonb(blob.meta) if blob and blob.meta else None
            db_document["blob_mime_type"] = blob.mime_type if blob and blob.mime_type else None

            db_document["dataframe"] = Jsonb(db_document["dataframe"]) if db_document["dataframe"] else None
            db_document["meta"] = Jsonb(db_document["meta"])

            if "sparse_embedding" in db_document:
                sparse_embedding = db_document.pop("sparse_embedding", None)
                if sparse_embedding:
                    logger.warning(
                        "Document %s has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in Pgvector is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        db_document["id"],
                    )

            db_documents.append(db_document)

        return db_documents

    @staticmethod
    def _from_pg_to_haystack_documents(documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Internal method to convert a list of dictionaries from pgvector to a list of Haystack Documents.
        """

        haystack_documents = []
        for document in documents:
            haystack_dict = dict(document)
            blob_data = haystack_dict.pop("blob_data")
            blob_meta = haystack_dict.pop("blob_meta")
            blob_mime_type = haystack_dict.pop("blob_mime_type")

            # postgresql returns the embedding as a string
            # so we need to convert it to a list of floats
            if document.get("embedding"):
                haystack_dict["embedding"] = [float(el) for el in document["embedding"].strip("[]").split(",")]

            haystack_document = Document.from_dict(haystack_dict)

            if blob_data:
                blob = ByteStream(data=blob_data, meta=blob_meta, mime_type=blob_mime_type)
                haystack_document.blob = blob

            haystack_documents.append(haystack_document)

        return haystack_documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """

        if not document_ids:
            return

        document_ids_str = ", ".join(f"'{document_id}'" for document_id in document_ids)

        delete_sql = SQL("DELETE FROM {table_name} WHERE id IN ({document_ids_str})").format(
            table_name=Identifier(self.table_name), document_ids_str=SQL(document_ids_str)
        )

        self._execute_sql(delete_sql, error_msg="Could not delete documents from PgvectorDocumentStore")

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]] = None,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        This method is not meant to be part of the public interface of
        `PgvectorDocumentStore` and it should not be called directly.
        `PgvectorEmbeddingRetriever` uses this method directly and is the public interface for it.
        :returns: List of Documents that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        if len(query_embedding) != self.embedding_dimension:
            msg = (
                f"query_embedding dimension ({len(query_embedding)}) does not match PgvectorDocumentStore "
                f"embedding dimension ({self.embedding_dimension})."
            )
            raise ValueError(msg)

        vector_function = vector_function or self.vector_function
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)

        # the vector must be a string with this format: "'[3,1,2]'"
        query_embedding_for_postgres = f"'[{','.join(str(el) for el in query_embedding)}]'"

        # to compute the scores, we use the approach described in pgvector README:
        # https://github.com/pgvector/pgvector?tab=readme-ov-file#distances
        # cosine_similarity and inner_product are modified from the result of the operator
        if vector_function == "cosine_similarity":
            score_definition = f"1 - (embedding <=> {query_embedding_for_postgres}) AS score"
        elif vector_function == "inner_product":
            score_definition = f"(embedding <#> {query_embedding_for_postgres}) * -1 AS score"
        elif vector_function == "l2_distance":
            score_definition = f"embedding <-> {query_embedding_for_postgres} AS score"

        sql_select = SQL("SELECT *, {score} FROM {table_name}").format(
            table_name=Identifier(self.table_name),
            score=SQL(score_definition),
        )

        sql_where_clause = SQL("")
        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)

        # we always want to return the most similar documents first
        # so when using l2_distance, the sort order must be ASC
        sort_order = "ASC" if vector_function == "l2_distance" else "DESC"

        sql_sort = SQL(" ORDER BY score {sort_order} LIMIT {top_k}").format(
            top_k=SQLLiteral(top_k),
            sort_order=SQL(sort_order),
        )

        sql_query = sql_select + sql_where_clause + sql_sort

        result = self._execute_sql(
            sql_query,
            params,
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
            cursor=self._dict_cursor,
        )

        records = result.fetchall()
        docs = self._from_pg_to_haystack_documents(records)
        return docs
