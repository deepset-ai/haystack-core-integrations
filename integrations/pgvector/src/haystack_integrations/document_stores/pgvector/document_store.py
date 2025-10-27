# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from psycopg import AsyncConnection, Connection, Cursor, Error, IntegrityError
from psycopg.cursor_async import AsyncCursor
from psycopg.rows import DictRow, dict_row
from psycopg.sql import SQL, Composed, Identifier
from psycopg.sql import Literal as SQLLiteral

from pgvector.psycopg import register_vector, register_vector_async

from .converters import _from_haystack_to_pg_documents, _from_pg_to_haystack_documents
from .filters import _convert_filters_to_where_clause_and_params, _validate_filters

logger = logging.getLogger(__name__)

CREATE_TABLE_STATEMENT = """
CREATE TABLE {schema_name}.{table_name} (
id VARCHAR(128) PRIMARY KEY,
embedding {embedding_col_type}({embedding_dimension}),
content TEXT,
blob_data BYTEA,
blob_meta JSONB,
blob_mime_type VARCHAR(255),
meta JSONB)
"""

INSERT_STATEMENT = """
INSERT INTO {schema_name}.{table_name}
(id, embedding, content, blob_data, blob_meta, blob_mime_type, meta)
VALUES (%(id)s, %(embedding)s, %(content)s, %(blob_data)s, %(blob_meta)s, %(blob_mime_type)s, %(meta)s)
"""

UPDATE_STATEMENT = """
ON CONFLICT (id) DO UPDATE SET
embedding = EXCLUDED.embedding,
content = EXCLUDED.content,
blob_data = EXCLUDED.blob_data,
blob_meta = EXCLUDED.blob_meta,
blob_mime_type = EXCLUDED.blob_mime_type,
meta = EXCLUDED.meta
"""

KEYWORD_QUERY = """
SELECT {table_name}.*, ts_rank_cd(to_tsvector({language}, content), query) AS score
FROM {schema_name}.{table_name}, plainto_tsquery({language}, %s) query
WHERE to_tsvector({language}, content) @@ query
"""

VALID_VECTOR_FUNCTIONS = ["cosine_similarity", "inner_product", "l2_distance"]

VECTOR_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_similarity": "vector_cosine_ops",
    "inner_product": "vector_ip_ops",
    "l2_distance": "vector_l2_ops",
}

HALF_VECTOR_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_similarity": "halfvec_cosine_ops",
    "inner_product": "halfvec_ip_ops",
    "l2_distance": "halfvec_l2_ops",
}

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]


class PgvectorDocumentStore:
    """
    A Document Store using PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector) installed.
    """

    def __init__(
        self,
        *,
        connection_string: Secret = Secret.from_env_var("PG_CONN_STR"),
        create_extension: bool = True,
        schema_name: str = "public",
        table_name: str = "haystack_documents",
        language: str = "english",
        embedding_dimension: int = 768,
        vector_type: Literal["vector", "halfvec"] = "vector",
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] = "cosine_similarity",
        recreate_table: bool = False,
        search_strategy: Literal["exact_nearest_neighbor", "hnsw"] = "exact_nearest_neighbor",
        hnsw_recreate_index_if_exists: bool = False,
        hnsw_index_creation_kwargs: Optional[Dict[str, int]] = None,
        hnsw_index_name: str = "haystack_hnsw_index",
        hnsw_ef_search: Optional[int] = None,
        keyword_index_name: str = "haystack_keyword_index",
    ):
        """
        Creates a new PgvectorDocumentStore instance.
        It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
        A specific table to store Haystack documents will be created if it doesn't exist yet.

        :param connection_string: The connection string to use to connect to the PostgreSQL database, defined as an
            environment variable. It can be provided in either URI format
            e.g.: `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"`, or keyword/value format
            e.g.: `PG_CONN_STR="host=HOST port=PORT dbname=DBNAME user=USER password=PASSWORD"`
            See [PostgreSQL Documentation](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
            for more details.
        :param create_extension: Whether to create the pgvector extension if it doesn't exist.
            Set this to `True` (default) to automatically create the extension if it is missing.
            Creating the extension may require superuser privileges.
            If set to `False`, ensure the extension is already installed; otherwise, an error will be raised.
        :param schema_name: The name of the schema the table is created in. The schema must already exist.
        :param table_name: The name of the table to use to store Haystack documents.
        :param language: The language to be used to parse query and document content in keyword retrieval.
            To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
            `SELECT cfgname FROM pg_ts_config;`.
            More information can be found in this [StackOverflow answer](https://stackoverflow.com/a/39752553).
        :param embedding_dimension: The dimension of the embedding.
        :param vector_type: The type of vector used for embedding storage.
            "vector" is the default.
            "halfvec" stores embeddings in half-precision, which is particularly useful for high-dimensional embeddings
            (dimension greater than 2,000 and up to 4,000). Requires pgvector versions 0.7.0 or later. For more
            information, see the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file).
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
        :param hnsw_index_name: Index name for the HNSW index.
        :param hnsw_ef_search: The `ef_search` parameter to use at query time. Only used if search_strategy is set to
            `"hnsw"`. You can find more information about this parameter in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
        :param keyword_index_name: Index name for the Keyword index.
        """

        self.connection_string = connection_string
        self.create_extension = create_extension
        self.table_name = table_name
        self.schema_name = schema_name
        self.embedding_dimension = embedding_dimension
        if vector_type not in ["vector", "halfvec"]:
            msg = "vector_type must be one of ['vector', 'halfvec']"
            raise ValueError(msg)
        self.vector_type = vector_type
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)
        self.vector_function = vector_function
        self.recreate_table = recreate_table
        self.search_strategy = search_strategy
        self.hnsw_recreate_index_if_exists = hnsw_recreate_index_if_exists
        self.hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}
        self.hnsw_index_name = hnsw_index_name
        self.hnsw_ef_search = hnsw_ef_search
        self.keyword_index_name = keyword_index_name
        self.language = language

        self._connection: Optional[Connection] = None
        self._async_connection: Optional[AsyncConnection] = None
        self._cursor: Optional[Cursor] = None
        self._async_cursor: Optional[AsyncCursor] = None
        self._dict_cursor: Optional[Cursor[DictRow]] = None
        self._async_dict_cursor: Optional[AsyncCursor[DictRow]] = None
        self._table_initialized = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_string=self.connection_string.to_dict(),
            create_extension=self.create_extension,
            schema_name=self.schema_name,
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            vector_type=self.vector_type,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
            search_strategy=self.search_strategy,
            hnsw_recreate_index_if_exists=self.hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=self.hnsw_index_creation_kwargs,
            hnsw_index_name=self.hnsw_index_name,
            hnsw_ef_search=self.hnsw_ef_search,
            keyword_index_name=self.keyword_index_name,
            language=self.language,
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

    @staticmethod
    def _connection_is_valid(connection):
        """
        Internal method to check if the connection is still valid.
        """

        # implementation inspired to psycopg pool
        # https://github.com/psycopg/psycopg/blob/d38cf7798b0c602ff43dac9f20bbab96237a9c38/psycopg_pool/psycopg_pool/pool.py#L528

        try:
            connection.execute("")
        except Error:
            return False
        return True

    @staticmethod
    async def _connection_is_valid_async(connection):
        """
        Internal method to check if the async connection is still valid.
        """
        try:
            await connection.execute("")
        except Error:
            return False
        return True

    @overload
    def _execute_sql(
        self, cursor: Cursor, sql_query: Composed, params: Optional[tuple] = None, error_msg: str = ""
    ) -> Cursor: ...

    @overload
    def _execute_sql(
        self, cursor: Cursor[DictRow], sql_query: Composed, params: Optional[tuple] = None, error_msg: str = ""
    ) -> Cursor[DictRow]: ...

    def _execute_sql(
        self,
        cursor: Union[Cursor, Cursor[DictRow]],
        sql_query: Composed,
        params: Optional[tuple] = None,
        error_msg: str = "",
    ) -> Union[Cursor, Cursor[DictRow]]:
        """
        Internal method to execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        :param cursor: The cursor to use to execute the SQL query.
        """

        params = params or ()

        if cursor is None or self._connection is None:
            message = (
                "The cursor or the connection is not initialized. "
                "Make sure to call _ensure_db_setup() before calling this method."
            )
            raise ValueError(message)

        sql_query_str = sql_query.as_string(cursor)
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=params)

        try:
            result = cursor.execute(sql_query, params)
        except Error as e:
            self._connection.rollback()
            detailed_error_msg = (
                f"{error_msg}. Error: {e!r}. \nYou can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(detailed_error_msg) from e

        return result

    @overload
    async def _execute_sql_async(
        self, cursor: AsyncCursor, sql_query: Composed, params: Optional[tuple] = None, error_msg: str = ""
    ) -> AsyncCursor: ...

    @overload
    async def _execute_sql_async(
        self, cursor: AsyncCursor[DictRow], sql_query: Composed, params: Optional[tuple] = None, error_msg: str = ""
    ) -> AsyncCursor[DictRow]: ...

    async def _execute_sql_async(
        self,
        cursor: Union[AsyncCursor, AsyncCursor[DictRow]],
        sql_query: Composed,
        params: Optional[tuple] = None,
        error_msg: str = "",
    ) -> Union[AsyncCursor, AsyncCursor[DictRow]]:
        """
        Internal method to asynchronously execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        :param cursor: The cursor to use to execute the SQL query.
        """

        params = params or ()

        if cursor is None or self._async_connection is None:
            message = (
                "The cursor or the connection is not initialized. "
                "Make sure to call _ensure_db_setup_async() before calling this method."
            )
            raise ValueError(message)

        sql_query_str = sql_query.as_string(cursor)
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=params)

        try:
            result = await cursor.execute(sql_query, params)
        except Error as e:
            await self._async_connection.rollback()
            detailed_error_msg = (
                f"{error_msg}. Error: {e!r}. \nYou can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(detailed_error_msg) from e

        return result

    def _ensure_db_setup(self):
        """
        Ensures that the connection to the PostgreSQL database exists and is valid.
        If not, connection and cursors are created.
        If the table is not initialized, it will be set up.
        """
        if self._connection and self._cursor and self._dict_cursor and self._connection_is_valid(self._connection):
            return

        # close the connection if it already exists
        if self._connection:
            try:
                self._connection.close()
            except Error as e:
                logger.debug("Failed to close connection: {e}", e=str(e))

        conn_str = self.connection_string.resolve_value() or ""
        connection = Connection.connect(conn_str)
        connection.autocommit = True
        if self.create_extension:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)  # Note: this must be called before creating the cursors.

        self._connection = connection
        self._cursor = self._connection.cursor()
        self._dict_cursor = self._connection.cursor(row_factory=dict_row)

        if not self._table_initialized:
            self._initialize_table()

    async def _ensure_db_setup_async(self):
        """
        Async internal method.
        Ensures that the connection to the PostgreSQL database exists and is valid.
        If not, connection and cursors are created.
        If the table is not initialized, it will be set up.
        """

        if (
            self._async_connection
            and self._async_cursor
            and self._async_dict_cursor
            and await self._connection_is_valid_async(self._async_connection)
        ):
            return

        # close the connection if it already exists
        if self._async_connection:
            await self._async_connection.close()

        conn_str = self.connection_string.resolve_value() or ""
        async_connection = await AsyncConnection.connect(conn_str)
        await async_connection.set_autocommit(True)
        if self.create_extension:
            await async_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector_async(async_connection)  # Note: this must be called before creating the cursors.

        self._async_connection = async_connection
        self._async_cursor = self._async_connection.cursor()
        self._async_dict_cursor = self._async_connection.cursor(row_factory=dict_row)

        if not self._table_initialized:
            await self._initialize_table_async()

    def _build_table_creation_queries(self):
        """
        Internal method to build the SQL queries for table creation.
        """

        sql_table_exists = SQL("SELECT 1 FROM pg_tables WHERE schemaname = %s AND tablename = %s")
        sql_create_table = SQL(CREATE_TABLE_STATEMENT).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            embedding_dimension=SQLLiteral(self.embedding_dimension),
            embedding_col_type=SQL(self.vector_type),
        )

        sql_keyword_index_exists = SQL(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s"
        )
        sql_create_keyword_index = SQL(
            "CREATE INDEX {index_name} ON {schema_name}.{table_name} USING GIN (to_tsvector({language}, content))"
        ).format(
            schema_name=Identifier(self.schema_name),
            index_name=Identifier(self.keyword_index_name),
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
        )

        return sql_table_exists, sql_create_table, sql_keyword_index_exists, sql_create_keyword_index

    def _initialize_table(self):
        """
        Internal method to initialize the table.
        """
        if self.recreate_table:
            self.delete_table()

        sql_table_exists, sql_create_table, sql_keyword_index_exists, sql_create_keyword_index = (
            self._build_table_creation_queries()
        )

        assert self._cursor is not None

        table_exists = bool(
            self._execute_sql(
                cursor=self._cursor,
                sql_query=sql_table_exists,
                params=(self.schema_name, self.table_name),
                error_msg="Could not check if table exists",
            ).fetchone()
        )
        if not table_exists:
            self._execute_sql(cursor=self._cursor, sql_query=sql_create_table, error_msg="Could not create table")

        index_exists = bool(
            self._execute_sql(
                cursor=self._cursor,
                sql_query=sql_keyword_index_exists,
                params=(self.schema_name, self.table_name, self.keyword_index_name),
                error_msg="Could not check if keyword index exists",
            ).fetchone()
        )
        if not index_exists:
            self._execute_sql(
                cursor=self._cursor,
                sql_query=sql_create_keyword_index,
                error_msg="Could not create keyword index on table",
            )

        if self.search_strategy == "hnsw":
            self._handle_hnsw()

        self._table_initialized = True

    async def _initialize_table_async(self):
        """
        Internal async method to initialize the table.
        """
        if self.recreate_table:
            await self.delete_table_async()

        sql_table_exists, sql_create_table, sql_keyword_index_exists, sql_create_keyword_index = (
            self._build_table_creation_queries()
        )

        assert self._async_cursor is not None

        table_exists = bool(
            await (
                await self._execute_sql_async(
                    cursor=self._async_cursor,
                    sql_query=sql_table_exists,
                    params=(self.schema_name, self.table_name),
                    error_msg="Could not check if table exists",
                )
            ).fetchone()
        )
        if not table_exists:
            await self._execute_sql_async(
                cursor=self._async_cursor, sql_query=sql_create_table, error_msg="Could not create table"
            )

        index_exists = bool(
            await (
                await self._execute_sql_async(
                    cursor=self._async_cursor,
                    sql_query=sql_keyword_index_exists,
                    params=(self.schema_name, self.table_name, self.keyword_index_name),
                    error_msg="Could not check if keyword index exists",
                )
            ).fetchone()
        )
        if not index_exists:
            await self._execute_sql_async(
                cursor=self._async_cursor,
                sql_query=sql_create_keyword_index,
                error_msg="Could not create keyword index on table",
            )

        if self.search_strategy == "hnsw":
            await self._handle_hnsw_async()

        self._table_initialized = True

    def delete_table(self):
        """
        Deletes the table used to store Haystack documents.
        The name of the schema (`schema_name`) and the name of the table (`table_name`)
        are defined when initializing the `PgvectorDocumentStore`.
        """
        self._ensure_db_setup()
        delete_sql = SQL("DROP TABLE IF EXISTS {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        assert self._cursor is not None

        self._execute_sql(
            cursor=self._cursor,
            sql_query=delete_sql,
            error_msg=f"Could not delete table {self.schema_name}.{self.table_name} in PgvectorDocumentStore",
        )

    async def delete_table_async(self):
        """
        Async method to delete the table used to store Haystack documents.
        """
        await self._ensure_db_setup_async()
        delete_sql = SQL("DROP TABLE IF EXISTS {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        assert self._async_cursor is not None

        await self._execute_sql_async(
            cursor=self._async_cursor,
            sql_query=delete_sql,
            error_msg=f"Could not delete table {self.schema_name}.{self.table_name} in PgvectorDocumentStore",
        )

    def _build_hnsw_queries(self):
        """Common method to build all HNSW-related SQL queries"""

        sql_set_hnsw_ef_search = (
            SQL("SET hnsw.ef_search = {hnsw_ef_search}").format(hnsw_ef_search=SQLLiteral(self.hnsw_ef_search))
            if self.hnsw_ef_search
            else None
        )

        sql_hnsw_index_exists = SQL(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s"
        )

        sql_drop_hnsw_index = SQL("DROP INDEX IF EXISTS {schema_name}.{index_name}").format(
            schema_name=Identifier(self.schema_name),
            index_name=Identifier(self.hnsw_index_name),
        )

        if self.vector_type == "halfvec":
            if self.vector_function not in HALF_VECTOR_FUNCTION_TO_POSTGRESQL_OPS:
                msg = f"Unsupported vector_function '{self.vector_function}' for halfvec type."
                raise ValueError(msg)
            pg_ops = HALF_VECTOR_FUNCTION_TO_POSTGRESQL_OPS[self.vector_function]
        else:
            if self.vector_function not in VECTOR_FUNCTION_TO_POSTGRESQL_OPS:
                msg = f"Unsupported vector_function '{self.vector_function}' for vector type."
                raise ValueError(msg)
            pg_ops = VECTOR_FUNCTION_TO_POSTGRESQL_OPS[self.vector_function]

        sql_create_hnsw_index = SQL(
            "CREATE INDEX {index_name} ON {schema_name}.{table_name} USING hnsw (embedding {ops})"
        ).format(
            schema_name=Identifier(self.schema_name),
            index_name=Identifier(self.hnsw_index_name),
            table_name=Identifier(self.table_name),
            ops=SQL(pg_ops),
        )

        # Add creation kwargs if any valid ones exist
        valid_kwargs = {
            k: v for k, v in self.hnsw_index_creation_kwargs.items() if k in HNSW_INDEX_CREATION_VALID_KWARGS
        }
        if valid_kwargs:
            kwargs_str = ", ".join(f"{k} = {v}" for k, v in valid_kwargs.items())
            sql_create_hnsw_index += SQL(" WITH ({})").format(SQL(kwargs_str))

        return sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index

    def _handle_hnsw(self):
        """
        Internal method to handle the HNSW index creation.
        It also sets the `hnsw.ef_search` parameter for queries if it is specified.
        """

        sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index = (
            self._build_hnsw_queries()
        )

        assert self._cursor is not None

        if self.hnsw_ef_search:
            self._execute_sql(
                cursor=self._cursor, sql_query=sql_set_hnsw_ef_search, error_msg="Could not set hnsw.ef_search"
            )

        index_exists = bool(
            self._execute_sql(
                cursor=self._cursor,
                sql_query=sql_hnsw_index_exists,
                params=(self.schema_name, self.table_name, self.hnsw_index_name),
                error_msg="Could not check if HNSW index exists",
            ).fetchone()
        )

        if index_exists and not self.hnsw_recreate_index_if_exists:
            logger.warning(
                "HNSW index already exists and won't be recreated. "
                "If you want to recreate it, pass 'hnsw_recreate_index_if_exists=True' to the "
                "Document Store constructor"
            )
            return

        self._execute_sql(cursor=self._cursor, sql_query=sql_drop_hnsw_index, error_msg="Could not drop HNSW index")

        self._execute_sql(cursor=self._cursor, sql_query=sql_create_hnsw_index, error_msg="Could not create HNSW index")

    async def _handle_hnsw_async(self):
        """
        Internal async method to handle the HNSW index creation.
        """

        sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index = (
            self._build_hnsw_queries()
        )

        assert self._async_cursor is not None

        if self.hnsw_ef_search:
            await self._execute_sql_async(
                cursor=self._async_cursor, sql_query=sql_set_hnsw_ef_search, error_msg="Could not set hnsw.ef_search"
            )

        index_exists = bool(
            await (
                await self._execute_sql_async(
                    cursor=self._async_cursor,
                    sql_query=sql_hnsw_index_exists,
                    params=(self.schema_name, self.table_name, self.hnsw_index_name),
                    error_msg="Could not check if HNSW index exists",
                )
            ).fetchone()
        )

        if index_exists and not self.hnsw_recreate_index_if_exists:
            logger.warning(
                "HNSW index already exists and won't be recreated. "
                "If you want to recreate it, pass 'hnsw_recreate_index_if_exists=True' to the "
                "Document Store constructor"
            )
            return

        await self._execute_sql_async(
            cursor=self._async_cursor, sql_query=sql_drop_hnsw_index, error_msg="Could not drop HNSW index"
        )

        await self._execute_sql_async(
            cursor=self._async_cursor, sql_query=sql_create_hnsw_index, error_msg="Could not create HNSW index"
        )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        self._ensure_db_setup()
        assert self._cursor is not None
        result = self._execute_sql(
            cursor=self._cursor, sql_query=sql_count, error_msg="Could not count documents in PgvectorDocumentStore"
        ).fetchone()
        if result is not None:
            return result[0]
        return 0

    async def count_documents_async(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None
        result = await (
            await self._execute_sql_async(
                cursor=self._async_cursor,
                sql_query=sql_count,
                error_msg="Could not count documents in PgvectorDocumentStore",
            )
        ).fetchone()

        if result is not None:
            return result[0]
        return 0

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :raises TypeError: If `filters` is not a dictionary.
        :raises ValueError: If `filters` syntax is invalid.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)

        sql_filter = SQL("SELECT * FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_filter += sql_where_clause

        self._ensure_db_setup()
        assert self._dict_cursor is not None

        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_filter,
            params=params,
            error_msg="Could not filter documents from PgvectorDocumentStore.",
        )

        records = result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.

        :raises TypeError: If `filters` is not a dictionary.
        :raises ValueError: If `filters` syntax is invalid.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)

        sql_filter = SQL("SELECT * FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_filter += sql_where_clause

        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None

        result = await self._execute_sql_async(
            cursor=self._async_dict_cursor,
            sql_query=sql_filter,
            params=params,
            error_msg="Could not filter documents from PgvectorDocumentStore.",
        )

        records = await result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    def _build_insert_statement(self, policy: DuplicatePolicy) -> Composed:
        """
        Builds the SQL insert statement to write documents.
        """
        sql_insert = SQL(INSERT_STATEMENT).format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        if policy == DuplicatePolicy.OVERWRITE:
            sql_insert += SQL(UPDATE_STATEMENT)
        elif policy == DuplicatePolicy.SKIP:
            sql_insert += SQL("ON CONFLICT DO NOTHING")

        sql_insert += SQL(" RETURNING id")

        return sql_insert

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises ValueError: If `documents` contains objects that are not of type `Document`.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :raises DocumentStoreError: If the write operation fails for any other reason.
        :returns: The number of documents written to the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        db_documents = _from_haystack_to_pg_documents(documents)

        sql_insert = self._build_insert_statement(policy)

        self._ensure_db_setup()
        assert self._cursor is not None  # verified in _ensure_db_setup() but mypy doesn't know that
        assert self._connection is not None  # verified in _ensure_db_setup() but mypy doesn't know that

        sql_query_str = sql_insert.as_string(self._cursor) if not isinstance(sql_insert, str) else sql_insert
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=db_documents)

        try:
            self._cursor.executemany(sql_insert, db_documents, returning=True)
        except IntegrityError as ie:
            self._connection.rollback()
            raise DuplicateDocumentError from ie
        except Error as e:
            self._connection.rollback()
            error_msg = (
                f"Could not write documents to PgvectorDocumentStore. Error: {e!r}. \n"
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

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Asynchronously writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises ValueError: If `documents` contains objects that are not of type `Document`.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :raises DocumentStoreError: If the write operation fails for any other reason.
        :returns: The number of documents written to the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        db_documents = _from_haystack_to_pg_documents(documents)

        sql_insert = self._build_insert_statement(policy)

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None  # verified in _ensure_db_setup_async() but mypy doesn't know that
        assert self._async_connection is not None  # verified in _ensure_db_setup_async() but mypy doesn't know that

        sql_query_str = sql_insert.as_string(self._async_cursor) if not isinstance(sql_insert, str) else sql_insert
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=db_documents)

        try:
            await self._async_cursor.executemany(sql_insert, db_documents, returning=True)
        except IntegrityError as ie:
            await self._async_connection.rollback()
            raise DuplicateDocumentError from ie
        except Error as e:
            await self._async_connection.rollback()
            error_msg = (
                f"Could not write documents to PgvectorDocumentStore. Error: {e!r}. \n"
                "You can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(error_msg) from e

        written_docs = 0
        async for _ in self._async_cursor:
            written_docs += 1

        return written_docs

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return

        document_ids_str = ", ".join(f"'{document_id}'" for document_id in document_ids)

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name} WHERE id IN ({document_ids_str})").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            document_ids_str=SQL(document_ids_str),
        )

        self._ensure_db_setup()
        assert self._cursor is not None
        self._execute_sql(
            cursor=self._cursor, sql_query=delete_sql, error_msg="Could not delete documents from PgvectorDocumentStore"
        )

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return

        document_ids_str = ", ".join(f"'{document_id}'" for document_id in document_ids)

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name} WHERE id IN ({document_ids_str})").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            document_ids_str=SQL(document_ids_str),
        )

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None
        await self._execute_sql_async(
            cursor=self._async_cursor,
            sql_query=delete_sql,
            error_msg="Could not delete documents from PgvectorDocumentStore",
        )

    def delete_all_documents(self) -> None:
        """
        Deletes all documents in the document store.
        """
        query = SQL("TRUNCATE TABLE {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        self._ensure_db_setup()
        assert self._cursor is not None
        self._execute_sql(
            cursor=self._cursor, sql_query=query, error_msg="Could not delete all documents from PgvectorDocumentStore"
        )

    async def delete_all_documents_async(self) -> None:
        """
        Asynchronously deletes all documents in the document store.
        """
        query = SQL("TRUNCATE TABLE {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None
        await self._execute_sql_async(
            cursor=self._async_cursor,
            sql_query=query,
            error_msg="Could not delete all documents from PgvectorDocumentStore",
        )

    def _build_keyword_retrieval_query(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[Composed, tuple]:
        """
        Builds the SQL query and the where parameters for keyword retrieval.
        """
        sql_select = SQL(KEYWORD_QUERY).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
            query=SQLLiteral(query),
        )

        where_params = ()
        sql_where_clause: Union[Composed, SQL] = SQL("")
        if filters:
            sql_where_clause, where_params = _convert_filters_to_where_clause_and_params(
                filters=filters, operator="AND"
            )

        sql_sort = SQL(" ORDER BY score DESC LIMIT {top_k}").format(top_k=SQLLiteral(top_k))

        sql_query = sql_select + sql_where_clause + sql_sort

        return sql_query, where_params

    def _keyword_retrieval(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query using a full-text search.

        This method is not meant to be part of the public interface of
        `PgvectorDocumentStore` and it should not be called directly.
        `PgvectorKeywordRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that are most similar to `query`
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        sql_query, where_params = self._build_keyword_retrieval_query(query=query, top_k=top_k, filters=filters)

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=(query, *where_params),
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
        )

        records = result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    async def _keyword_retrieval_async(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query using a full-text search asynchronously.
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        sql_query, where_params = self._build_keyword_retrieval_query(query=query, top_k=top_k, filters=filters)

        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None
        result = await self._execute_sql_async(
            cursor=self._async_dict_cursor,
            sql_query=sql_query,
            params=(query, *where_params),
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
        )

        records = await result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    def _check_and_build_embedding_retrieval_query(
        self,
        query_embedding: List[float],
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Composed, tuple]:
        """
        Performs checks and builds the SQL query and the where parameters for embedding retrieval.
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

        sql_select = SQL("SELECT *, {score} FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            score=SQL(score_definition),
        )

        sql_where_clause: Union[Composed, SQL] = SQL("")
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

        return sql_query, params

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

        sql_query, params = self._check_and_build_embedding_retrieval_query(
            query_embedding=query_embedding, vector_function=vector_function, top_k=top_k, filters=filters
        )
        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
        )

        records = result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    async def _embedding_retrieval_async(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]] = None,
    ) -> List[Document]:
        """
        Asynchronously retrieves documents that are most similar to the query embedding using a
        vector similarity metric.
        """

        sql_query, params = self._check_and_build_embedding_retrieval_query(
            query_embedding=query_embedding, vector_function=vector_function, top_k=top_k, filters=filters
        )

        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None
        result = await self._execute_sql_async(
            cursor=self._async_dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
        )

        records = await result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs
