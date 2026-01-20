# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, overload

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
from psycopg.types.json import Jsonb

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
        hnsw_index_creation_kwargs: dict[str, int] | None = None,
        hnsw_index_name: str = "haystack_hnsw_index",
        hnsw_ef_search: int | None = None,
        keyword_index_name: str = "haystack_keyword_index",
    ):
        """
        Creates a new PgvectorDocumentStore instance.
        It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
        A specific table to store Haystack documents will be created if it doesn't exist yet.

        :param connection_string: The connection string to use to connect to the PostgreSQL database, defined as an
            environment variable. Supported formats:
            - URI, e.g. `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"` (use percent-encoding for special
                characters)
            - keyword/value format, e.g. `PG_CONN_STR="host=HOST port=PORT dbname=DBNAME user=USER password=PASSWORD"`
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

        self._connection: Connection | None = None
        self._async_connection: AsyncConnection | None = None
        self._cursor: Cursor | None = None
        self._async_cursor: AsyncCursor | None = None
        self._dict_cursor: Cursor[DictRow] | None = None
        self._async_dict_cursor: AsyncCursor[DictRow] | None = None
        self._table_initialized = False

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "PgvectorDocumentStore":
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
        self, cursor: Cursor, sql_query: Composed, params: tuple | None = None, error_msg: str = ""
    ) -> Cursor: ...

    @overload
    def _execute_sql(
        self, cursor: Cursor[DictRow], sql_query: Composed, params: tuple | None = None, error_msg: str = ""
    ) -> Cursor[DictRow]: ...

    def _execute_sql(
        self,
        cursor: Cursor | Cursor[DictRow],
        sql_query: Composed,
        params: tuple | None = None,
        error_msg: str = "",
    ) -> Cursor | Cursor[DictRow]:
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
        self, cursor: AsyncCursor, sql_query: Composed, params: tuple | None = None, error_msg: str = ""
    ) -> AsyncCursor: ...

    @overload
    async def _execute_sql_async(
        self, cursor: AsyncCursor[DictRow], sql_query: Composed, params: tuple | None = None, error_msg: str = ""
    ) -> AsyncCursor[DictRow]: ...

    async def _execute_sql_async(
        self,
        cursor: AsyncCursor | AsyncCursor[DictRow],
        sql_query: Composed,
        params: tuple | None = None,
        error_msg: str = "",
    ) -> AsyncCursor | AsyncCursor[DictRow]:
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
        try:
            connection = Connection.connect(conn_str)
        except Error as e:
            msg = (
                "Failed to connect to PostgreSQL database.  Ensure the connection string follows the "
                "PostgreSQL connection specification: "
                "https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING."
            )
            raise DocumentStoreError(msg) from e
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
        try:
            async_connection = await AsyncConnection.connect(conn_str)
        except Error as e:
            msg = (
                "Failed to connect to PostgreSQL database.  Ensure the connection string follows the "
                "PostgreSQL connection specification: "
                "https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING."
            )
            raise DocumentStoreError(msg) from e
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

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

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

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

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

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
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
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
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

    def delete_documents(self, document_ids: list[str]) -> None:
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

    async def delete_documents_async(self, document_ids: list[str]) -> None:
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

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        _validate_filters(filters)

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            delete_sql += sql_where_clause

        self._ensure_db_setup()
        assert self._cursor is not None

        try:
            self._execute_sql(
                cursor=self._cursor,
                sql_query=delete_sql,
                params=params,
                error_msg="Could not delete documents by filter from PgvectorDocumentStore",
            )
            deleted_count = self._cursor.rowcount
            logger.info(
                "Deleted {n_docs} documents from table '{schema}.{table}' using filters.",
                n_docs=deleted_count,
                schema=self.schema_name,
                table=self.table_name,
            )
            return deleted_count
        except Error as e:
            msg = f"Failed to delete documents by filter from PgvectorDocumentStore: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        _validate_filters(filters)

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            delete_sql += sql_where_clause

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None

        try:
            await self._execute_sql_async(
                cursor=self._async_cursor,
                sql_query=delete_sql,
                params=params,
                error_msg="Could not delete documents by filter from PgvectorDocumentStore",
            )
            deleted_count = self._async_cursor.rowcount
            logger.info(
                "Deleted {n_docs} documents from table '{schema}.{table}' using filters.",
                n_docs=deleted_count,
                schema=self.schema_name,
                table=self.table_name,
            )
            return deleted_count
        except Error as e:
            msg = f"Failed to delete documents by filter from PgvectorDocumentStore: {e!s}"
            raise DocumentStoreError(msg) from e

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :returns: The number of documents updated.
        """
        _validate_filters(filters)

        if not meta:
            msg = "meta must be a non-empty dictionary"
            raise ValueError(msg)

        update_sql = SQL(
            "UPDATE {schema_name}.{table_name} SET meta = COALESCE(meta, '{{}}'::jsonb) || %s::jsonb"
        ).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        params: tuple[Any, ...] = (Jsonb(meta),)
        if filters:
            sql_where_clause, where_params = _convert_filters_to_where_clause_and_params(filters)
            update_sql += sql_where_clause
            params = params + where_params

        self._ensure_db_setup()
        assert self._cursor is not None

        try:
            self._execute_sql(
                cursor=self._cursor,
                sql_query=update_sql,
                params=params,
                error_msg="Could not update documents by filter from PgvectorDocumentStore",
            )
            updated_count = self._cursor.rowcount
            logger.info(
                "Updated {n_docs} documents in table '{schema}.{table}' using filters.",
                n_docs=updated_count,
                schema=self.schema_name,
                table=self.table_name,
            )
            return updated_count
        except Error as e:
            msg = f"Failed to update documents by filter in PgvectorDocumentStore: {e!s}"
            raise DocumentStoreError(msg) from e

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Asynchronously updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :returns: The number of documents updated.
        """
        _validate_filters(filters)

        if not meta:
            msg = "meta must be a non-empty dictionary"
            raise ValueError(msg)

        update_sql = SQL(
            "UPDATE {schema_name}.{table_name} SET meta = COALESCE(meta, '{{}}'::jsonb) || %s::jsonb"
        ).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        params: tuple[Any, ...] = (Jsonb(meta),)
        if filters:
            sql_where_clause, where_params = _convert_filters_to_where_clause_and_params(filters)
            update_sql += sql_where_clause
            params = params + where_params

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None

        try:
            await self._execute_sql_async(
                cursor=self._async_cursor,
                sql_query=update_sql,
                params=params,
                error_msg="Could not update documents by filter from PgvectorDocumentStore",
            )
            updated_count = self._async_cursor.rowcount
            logger.info(
                "Updated {n_docs} documents in table '{schema}.{table}' using filters.",
                n_docs=updated_count,
                schema=self.schema_name,
                table=self.table_name,
            )
            return updated_count
        except Error as e:
            msg = f"Failed to update documents by filter in PgvectorDocumentStore: {e!s}"
            raise DocumentStoreError(msg) from e

    def _build_keyword_retrieval_query(
        self, query: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> tuple[Composed, tuple]:
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
        sql_where_clause: Composed | SQL = SQL("")
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
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
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
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
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
        query_embedding: list[float],
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] | None,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[Composed, tuple]:
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

        sql_where_clause: Composed | SQL = SQL("")
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
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] | None = None,
    ) -> list[Document]:
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
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] | None = None,
    ) -> list[Document]:
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

    def _prepare_filters_count_documents(self, filters):
        _validate_filters(filters)
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )
        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_count += sql_where_clause
        return params, sql_count

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the number of documents that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents that match the filters.
        """
        params, sql_count = self._prepare_filters_count_documents(filters)

        self._ensure_db_setup()
        assert self._cursor is not None
        result = self._execute_sql(
            cursor=self._cursor,
            sql_query=sql_count,
            params=params,
            error_msg="Could not count documents by filter in PgvectorDocumentStore",
        ).fetchone()

        if result is not None:
            return result[0]
        return 0

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously returns the number of documents that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents that match the filters.
        """
        params, sql_count = self._prepare_filters_count_documents(filters)

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None
        result = await (
            await self._execute_sql_async(
                cursor=self._async_cursor,
                sql_query=sql_count,
                params=params,
                error_msg="Could not count documents by filter in PgvectorDocumentStore",
            )
        ).fetchone()

        if result is not None:
            return result[0]
        return 0

    @staticmethod
    def _normalize_metadata_field_name(field_name: str) -> str:
        """
        Normalizes metadata field names by removing 'meta.' prefix if present.

        :param field_name: The field name to normalize.
        :returns: The normalized field name.
        """
        if field_name.startswith("meta."):
            field_name = field_name[5:]  # Remove "meta." prefix

        # Validate field name to prevent SQL injection
        # Only allow alphanumeric characters, underscores, and hyphens
        if not all(c.isalnum() or c in ("_", "-") for c in field_name):
            msg = (f"Invalid metadata field name: '{field_name}'. Field names can only contain alphanumeric "
                   f"characters, underscores, and hyphens.")
            raise ValueError(msg)

        return field_name

    def _build_count_unique_metadata_query(
        self, normalized_fields: list[str], filters: dict[str, Any]
    ) -> tuple[Composed, tuple]:
        """
        Builds the SQL query for counting unique metadata values.

        :param normalized_fields: List of normalized metadata field names.
        :param filters: The filters to apply to select documents.
        :returns: A tuple containing (sql_query, params).
        """
        # Build SELECT clause with COUNT(DISTINCT ...) for each field
        count_expressions = []
        for field in normalized_fields:
            # Use SQLLiteral for the JSONB key (validated field name)
            count_expressions.append(
                SQL("COUNT(DISTINCT meta->>{} ) AS {}").format(SQLLiteral(field), Identifier(field))
            )

        sql_select = SQL("SELECT ") + SQL(", ").join(count_expressions)
        sql_from = SQL(" FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        sql_query = sql_select + sql_from

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_query += sql_where_clause

        return sql_query, params

    @staticmethod
    def _process_count_unique_metadata_result(
        result: dict[str, Any] | None,
        normalized_fields: list[str]
    ) -> dict[str, int]:
        """
        Processes the result from counting unique metadata values.

        :param result: The database result row, or None if no results.
        :param normalized_fields: List of normalized metadata field names.
        :returns: A dictionary mapping field names to their unique value counts.
        """
        if result is None:
            return dict.fromkeys(normalized_fields, 0)

        # Return dictionary with normalized field names
        return {field: result.get(field, 0) for field in normalized_fields}

    def count_unique_metadata_by_filter(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Returns the count of unique values for each specified metadata field,
        considering only documents that match the provided filters.

        :param filters: The filters to apply to select documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param metadata_fields: List of metadata field names to count unique values for.
            Field names can include or omit the "meta." prefix.
        :returns: A dictionary mapping field names to their unique value counts.
        """
        _validate_filters(filters)

        if not metadata_fields:
            msg = "metadata_fields must be a non-empty list"
            raise ValueError(msg)

        normalized_fields = [PgvectorDocumentStore._normalize_metadata_field_name(field) for field in metadata_fields]
        sql_query, params = self._build_count_unique_metadata_query(normalized_fields, filters)

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg="Could not count unique metadata values in PgvectorDocumentStore",
        ).fetchone()

        return PgvectorDocumentStore._process_count_unique_metadata_result(result, normalized_fields)

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Asynchronously returns the count of unique values for each specified metadata field,
        considering only documents that match the provided filters.

        :param filters: The filters to apply to select documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param metadata_fields: List of metadata field names to count unique values for.
            Field names can include or omit the "meta." prefix.
        :returns: A dictionary mapping field names to their unique value counts.
        """
        _validate_filters(filters)

        if not metadata_fields:
            msg = "metadata_fields must be a non-empty list"
            raise ValueError(msg)

        normalized_fields = [PgvectorDocumentStore._normalize_metadata_field_name(field) for field in metadata_fields]
        sql_query, params = self._build_count_unique_metadata_query(normalized_fields, filters)

        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None
        result = await (
            await self._execute_sql_async(
                cursor=self._async_dict_cursor,
                sql_query=sql_query,
                params=params,
                error_msg="Could not count unique metadata values in PgvectorDocumentStore",
            )
        ).fetchone()

        return PgvectorDocumentStore._process_count_unique_metadata_result(result, normalized_fields)

    @staticmethod
    def _infer_metadata_field_type(value: Any) -> str:
        """
        Infers the PostgreSQL/JSONB type from a Python value.

        :param value: The value to infer the type from.
        :returns: The inferred type name (text, integer, real, boolean).
        """
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "real"
        if isinstance(value, str):
            return "text"
        return "text"  # Default fallback

    @staticmethod
    def _analyze_metadata_fields_from_records(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        """
        Analyzes metadata fields from database records and infers their types.

        :param records: List of database records containing 'meta' field.
        :returns: A dictionary mapping field names to their type information.
        """
        fields_info: dict[str, dict[str, str]] = {"content": {"type": "text"}}

        # Analyze metadata from all documents
        for record in records:
            meta = record.get("meta")
            if not isinstance(meta, dict):
                continue

            for field_name, field_value in meta.items():
                if field_name not in fields_info:
                    # Infer type from first non-null value encountered
                    if field_value is not None:
                        inferred_type = PgvectorDocumentStore._infer_metadata_field_type(field_value)
                        fields_info[field_name] = {"type": inferred_type}
                    else:
                        # Default to text for null values
                        fields_info[field_name] = {"type": "text"}

        return fields_info

    def _analyze_metadata_from_docs_query(self):
        # query all documents to analyze metadata structure
        sql_query = SQL("SELECT meta FROM {schema_name}.{table_name} WHERE meta IS NOT NULL").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )
        return sql_query

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns the information about the metadata fields in the document store.

        Since metadata is stored in a JSONB field, this method analyzes actual data
        to infer field types.

        Example return:
        ```python
        {
            'content': {'type': 'text'},
            'category': {'type': 'text'},
            'status': {'type': 'text'},
            'priority': {'type': 'integer'},
        }
        ```

        :returns: A dictionary mapping field names to their type information.
        """
        self._ensure_db_setup()
        assert self._dict_cursor is not None

        sql_query = self._analyze_metadata_from_docs_query()

        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            error_msg="Could not retrieve metadata fields info from PgvectorDocumentStore",
        )

        records = result.fetchall()
        return PgvectorDocumentStore._analyze_metadata_fields_from_records(records)

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """
        Asynchronously returns the information about the metadata fields in the document store.

        Since metadata is stored in a JSONB field, this method analyzes actual data
        to infer field types.

        :returns: A dictionary mapping field names to their type information.
        """
        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None

        sql_query = self._analyze_metadata_from_docs_query()

        result = await self._execute_sql_async(
            cursor=self._async_dict_cursor,
            sql_query=sql_query,
            error_msg="Could not retrieve metadata fields info from PgvectorDocumentStore",
        )

        records = await result.fetchall()
        return PgvectorDocumentStore._analyze_metadata_fields_from_records(records)

    def _build_min_max_query(self, normalized_field: str, field_type: str) -> Composed:
        """
        Builds the SQL query for getting min/max values based on the field type.

        :param normalized_field: The normalized metadata field name.
        :param field_type: The type of the field (integer, real, text, or boolean).
        :returns: The SQL query for min/max calculation.
        """
        field_literal = SQLLiteral(normalized_field)

        if field_type == "integer":
            # For integer fields, cast directly to integer
            sql_query = SQL(
                """
                SELECT 
                    MIN((meta->>{} )::integer) AS min_value,
                    MAX((meta->>{} )::integer) AS max_value
                FROM {}.{}
                WHERE meta->>{} IS NOT NULL
                """ # noqa: W291
            ).format(
                field_literal,
                field_literal,
                Identifier(self.schema_name),
                Identifier(self.table_name),
                field_literal,
            )
        elif field_type == "real":
            # For real (float) fields, cast directly to real
            sql_query = SQL(
                """
                SELECT 
                    MIN((meta->>{} )::real) AS min_value,
                    MAX((meta->>{} )::real) AS max_value
                FROM {}.{}
                WHERE meta->>{} IS NOT NULL
                """ # noqa: W291
            ).format(
                field_literal,
                field_literal,
                Identifier(self.schema_name),
                Identifier(self.table_name),
                field_literal,
            )
        else:
            # For text and other non-numeric fields, use text comparison
            # Use COLLATE "C" for case-sensitive comparison (byte-order comparison)
            # This ensures uppercase and lowercase letters are treated differently
            sql_query = SQL(
                """
                SELECT 
                    MIN(meta->>{} COLLATE "C") AS min_value,
                    MAX(meta->>{} COLLATE "C") AS max_value
                FROM {}.{}
                WHERE meta->>{} IS NOT NULL
                """ # noqa: W291
            ).format(
                field_literal,
                field_literal,
                Identifier(self.schema_name),
                Identifier(self.table_name),
                field_literal,
            )

        return sql_query

    @staticmethod
    def _process_min_max_result(metadata_field: str, result: dict[str, Any] | None) -> tuple[Any, Any]:
        """
        Processes the result from min/max query.

        :param metadata_field: The metadata field name (for error messages).
        :param result: The database result row, or None if no results.
        :returns: A tuple containing (max_value, min_value).
        :raises ValueError: If the field has no values.
        """
        if result is None:
            msg = f"Metadata field '{metadata_field}' has no values"
            raise ValueError(msg)
        min_value = result.get("min_value")
        max_value = result.get("max_value")
        if min_value is None and max_value is None:
            msg = f"Metadata field '{metadata_field}' has no values"
            raise ValueError(msg)
        return max_value, min_value

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        Returns the minimum and maximum values for a given metadata field.

        :param metadata_field: The name of the metadata field. Can include or omit the "meta." prefix.
        :returns: A dictionary with 'min' and 'max' keys containing the minimum and maximum values.
            For numeric fields (integer, real), returns numeric min/max.
            For text fields, returns lexicographic min/max based on database collation.
        :raises ValueError: If the field doesn't exist or has no values.
        """
        normalized_field = PgvectorDocumentStore._normalize_metadata_field_name(metadata_field)

        # Get field type information from metadata fields info
        fields_info = self.get_metadata_fields_info()
        if normalized_field not in fields_info:
            msg = f"Metadata field '{metadata_field}' not found in document store"
            raise ValueError(msg)

        field_type = fields_info[normalized_field]["type"]
        sql_query = self._build_min_max_query(normalized_field, field_type)

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            error_msg=f"Could not get min/max for metadata field '{metadata_field}' in PgvectorDocumentStore",
        ).fetchone()

        max_value, min_value = PgvectorDocumentStore._process_min_max_result(metadata_field, result)

        return {"min": min_value, "max": max_value}

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, Any]:
        """
        Asynchronously returns the minimum and maximum values for a given metadata field.

        :param metadata_field: The name of the metadata field. Can include or omit the "meta." prefix.
        :returns: A dictionary with 'min' and 'max' keys containing the minimum and maximum values.
            For numeric fields (integer, real), returns numeric min/max.
            For text fields, returns lexicographic min/max based on database collation.
        :raises ValueError: If the field doesn't exist or has no values.
        """
        normalized_field = PgvectorDocumentStore._normalize_metadata_field_name(metadata_field)

        # Get field type information from metadata fields info
        fields_info = await self.get_metadata_fields_info_async()
        if normalized_field not in fields_info:
            msg = f"Metadata field '{metadata_field}' not found in document store"
            raise ValueError(msg)

        field_type = fields_info[normalized_field]["type"]
        sql_query = self._build_min_max_query(normalized_field, field_type)

        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None
        result = await (
            await self._execute_sql_async(
                cursor=self._async_dict_cursor,
                sql_query=sql_query,
                error_msg=f"Could not get min/max for metadata field '{metadata_field}' in PgvectorDocumentStore",
            )
        ).fetchone()

        max_value, min_value = PgvectorDocumentStore._process_min_max_result(metadata_field, result)

        return {"min": min_value, "max": max_value}

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: str | None, from_: int, size: int
    ) -> tuple[list[str], int]:
        """
        Returns unique values for a given metadata field, optionally filtered by a search term.

        :param metadata_field: The name of the metadata field. Can include or omit the "meta." prefix.
        :param search_term: Optional search term to filter documents by content before extracting unique values.
            If None, all documents are considered.
        :param from_: The offset for pagination (0-based).
        :param size: The number of unique values to return.
        :returns: A tuple containing:
            - A list of unique values (as strings)
            - The total count of unique values
        """
        normalized_field = PgvectorDocumentStore._normalize_metadata_field_name(metadata_field)

        # Field name is validated, so it's safe to use SQLLiteral
        field_literal = SQLLiteral(normalized_field)

        # Build the base query
        sql_select = SQL("SELECT DISTINCT meta->>{} AS value").format(field_literal)
        sql_from = SQL(" FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        sql_where = SQL(" WHERE meta->>{} IS NOT NULL").format(field_literal)

        params: tuple = ()
        if search_term:
            # Use full-text search with word boundaries (similar to keyword retrieval)
            # This matches OpenSearch's match_phrase behavior more closely
            sql_where += SQL(" AND to_tsvector({language}, content) @@ plainto_tsquery({language}, %s)").format(
                language=SQLLiteral(self.language)
            )
            params = (search_term,)

        # Get total count first
        sql_count = SQL("SELECT COUNT(DISTINCT meta->>{} ) AS total").format(field_literal)
        sql_count += sql_from + sql_where

        self._ensure_db_setup()
        assert self._dict_cursor is not None

        count_result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_count,
            params=params,
            error_msg=f"Could not count unique values for metadata field '{metadata_field}' in PgvectorDocumentStore",
        ).fetchone()

        total_count = count_result.get("total", 0) if count_result else 0

        # Get paginated unique values
        sql_query = sql_select + sql_from + sql_where
        sql_query += SQL(" ORDER BY value LIMIT {size} OFFSET {from_}").format(
            size=SQLLiteral(size), from_=SQLLiteral(from_)
        )

        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg=f"Could not get unique values for metadata field '{metadata_field}' in PgvectorDocumentStore",
        )

        records = result.fetchall()
        unique_values = [str(record.get("value", "")) for record in records if record.get("value") is not None]

        return unique_values, total_count

    async def get_metadata_field_unique_values_async(
        self, metadata_field: str, search_term: str | None, from_: int, size: int
    ) -> tuple[list[str], int]:
        """
        Asynchronously returns unique values for a given metadata field, optionally filtered by a search term.

        :param metadata_field: The name of the metadata field. Can include or omit the "meta." prefix.
        :param search_term: Optional search term to filter documents by content before extracting unique values.
            If None, all documents are considered.
        :param from_: The offset for pagination (0-based).
        :param size: The number of unique values to return.
        :returns: A tuple containing:
            - A list of unique values (as strings)
            - The total count of unique values
        """
        normalized_field = PgvectorDocumentStore._normalize_metadata_field_name(metadata_field)

        # Field name is validated, so it's safe to use SQLLiteral
        field_literal = SQLLiteral(normalized_field)

        # Build the base query
        sql_select = SQL("SELECT DISTINCT meta->>{} AS value").format(field_literal)
        sql_from = SQL(" FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        sql_where = SQL(" WHERE meta->>{} IS NOT NULL").format(field_literal)

        params: tuple = ()
        if search_term:
            # Use full-text search with word boundaries (similar to keyword retrieval)
            # This matches OpenSearch's match_phrase behavior more closely
            sql_where += SQL(" AND to_tsvector({language}, content) @@ plainto_tsquery({language}, %s)").format(
                language=SQLLiteral(self.language)
            )
            params = (search_term,)

        # Get total count first
        sql_count = SQL("SELECT COUNT(DISTINCT meta->>{} ) AS total").format(field_literal)
        sql_count += sql_from + sql_where

        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None

        count_result = await (
            await self._execute_sql_async(
                cursor=self._async_dict_cursor,
                sql_query=sql_count,
                params=params,
                error_msg=f"Could not count unique values for metadata field '{metadata_field}' in "
                          f"PgvectorDocumentStore",
            )
        ).fetchone()

        total_count = count_result.get("total", 0) if count_result else 0

        # Get paginated unique values
        sql_query = sql_select + sql_from + sql_where
        sql_query += SQL(" ORDER BY value LIMIT {size} OFFSET {from_}").format(
            size=SQLLiteral(size), from_=SQLLiteral(from_)
        )

        result = await self._execute_sql_async(
            cursor=self._async_dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg=f"Could not get unique values for metadata field '{metadata_field}' in PgvectorDocumentStore",
        )

        records = await result.fetchall()
        unique_values = [str(record.get("value", "")) for record in records if record.get("value") is not None]

        return unique_values, total_count

    def query_sql(self, query: str) -> Any:
        """
        Executes a raw SQL query against the document store.

        **Warning**: This method allows direct SQL execution. Use with caution and ensure
        queries are safe to prevent SQL injection. Prefer using the provided methods
        for document operations.

        :param query: The SQL query string to execute.
        :returns: The query result. The exact type depends on the query executed.
        :raises DocumentStoreError: If the query execution fails.
        """
        self._ensure_db_setup()
        assert self._dict_cursor is not None

        try:
            result = self._dict_cursor.execute(SQL(query))
            # Try to fetch results if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                return result.fetchall()
            # For other queries, return rowcount
            return {"rowcount": self._dict_cursor.rowcount}
        except Error as e:
            if self._connection:
                self._connection.rollback()
            error_msg = f"Failed to execute SQL query in PgvectorDocumentStore: {e!r}"
            raise DocumentStoreError(error_msg) from e

    async def query_sql_async(self, query: str) -> Any:
        """
        Asynchronously executes a raw SQL query against the document store.

        **Warning**: This method allows direct SQL execution. Use with caution and ensure
        queries are safe to prevent SQL injection. Prefer using the provided methods
        for document operations.

        :param query: The SQL query string to execute.
        :returns: The query result. The exact type depends on the query executed.
        :raises DocumentStoreError: If the query execution fails.
        """
        await self._ensure_db_setup_async()
        assert self._async_dict_cursor is not None

        try:
            result = await self._async_dict_cursor.execute(SQL(query))
            # Try to fetch results if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                return await result.fetchall()
            # For other queries, return rowcount
            return {"rowcount": self._async_dict_cursor.rowcount}
        except Error as e:
            if self._async_connection:
                await self._async_connection.rollback()
            error_msg = f"Failed to execute SQL query in PgvectorDocumentStore: {e!r}"
            raise DocumentStoreError(error_msg) from e
