# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, overload

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, Error, IntegrityError
from psycopg.rows import DictRow, dict_row
from psycopg.sql import SQL, Composed, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Jsonb

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

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]


class AlloyDBDocumentStore(DocumentStore):
    """
    A Document Store backed by [Google Cloud AlloyDB](https://cloud.google.com/alloydb).

    Uses the [pgvector extension](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings) for vector search.

    AlloyDB is a fully managed, PostgreSQL-compatible database service on Google Cloud.
    Connection is handled securely via the
    [AlloyDB Python Connector](https://github.com/GoogleCloudPlatform/alloydb-python-connector),
    which provides TLS encryption and IAM-based authorization without requiring manual SSL certificate
    management, firewall rules, or IP allowlisting.

    **Filter limitations**: the `NOT` logical operator is not supported. Use `!=` or `not in`
    comparison operators to express negation.

    Usage example:
    ```python
    import os
    from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore

    # Set required environment variables:
    # ALLOYDB_INSTANCE_URI = "projects/MY_PROJECT/locations/MY_REGION/clusters/MY_CLUSTER/instances/MY_INSTANCE"
    # ALLOYDB_USER = "my-db-user"
    # ALLOYDB_PASSWORD = "my-db-password"

    document_store = AlloyDBDocumentStore(
        db="my-database",
        embedding_dimension=768,
        recreate_table=True,
    )
    ```
    """

    def __init__(
        self,
        *,
        instance_uri: Secret = Secret.from_env_var("ALLOYDB_INSTANCE_URI"),
        user: Secret = Secret.from_env_var("ALLOYDB_USER"),
        password: Secret = Secret.from_env_var("ALLOYDB_PASSWORD", strict=False),
        db: str = "postgres",
        enable_iam_auth: bool = False,
        ip_type: Literal["PRIVATE", "PUBLIC", "PSC"] = "PRIVATE",
        create_extension: bool = True,
        schema_name: str = "public",
        table_name: str = "haystack_documents",
        language: str = "english",
        embedding_dimension: int = 768,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] = "cosine_similarity",
        recreate_table: bool = False,
        search_strategy: Literal["exact_nearest_neighbor", "hnsw"] = "exact_nearest_neighbor",
        hnsw_recreate_index_if_exists: bool = False,
        hnsw_index_creation_kwargs: dict[str, int] | None = None,
        hnsw_index_name: str = "haystack_hnsw_index",
        hnsw_ef_search: int | None = None,
        keyword_index_name: str = "haystack_keyword_index",
    ) -> None:
        """
        Creates a new AlloyDBDocumentStore instance.

        Connection to AlloyDB is established lazily on first use via the AlloyDB Python Connector.
        A specific table to store Haystack documents will be created if it doesn't exist yet.

        :param instance_uri: The AlloyDB instance URI in the format
            `"projects/PROJECT/locations/REGION/clusters/CLUSTER/instances/INSTANCE"`.
            Read from the `ALLOYDB_INSTANCE_URI` environment variable by default.
        :param user: The database user. Read from the `ALLOYDB_USER` environment variable by default.
            When using IAM database authentication, use the service account email (omitting
            `.gserviceaccount.com`) or the full IAM user email.
        :param password: The database password. Read from the `ALLOYDB_PASSWORD` environment variable by default.
            Not required when `enable_iam_auth=True`.
        :param db: The name of the database to connect to. Defaults to `"postgres"`.
        :param enable_iam_auth: Whether to use IAM database authentication instead of a password.
            When `True`, `password` is ignored. The IAM principal must be granted the
            AlloyDB Client role and have an IAM database user created.
            See the [AlloyDB documentation](https://cloud.google.com/alloydb/docs/manage-iam-authn) for details.
        :param ip_type: The IP address type to use for the connection.
            `"PRIVATE"` (default) connects over a private VPC IP.
            `"PUBLIC"` connects over a public IP.
            `"PSC"` connects via Private Service Connect.
        :param create_extension: Whether to create the pgvector extension if it doesn't exist.
            Set this to `True` (default) to automatically create the extension if it is missing.
            Creating the extension may require superuser privileges.
            If set to `False`, ensure the extension is already installed; otherwise, an error will be raised.
        :param schema_name: The name of the schema the table is created in. The schema must already exist.
        :param table_name: The name of the table to use to store Haystack documents.
        :param language: The language to be used to parse query and document content in keyword retrieval.
            To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
            `SELECT cfgname FROM pg_ts_config;`.
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
            Only used if search_strategy is set to `"hnsw"`. Valid arguments are `m` and `ef_construction`.
            See the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw) for details.
        :param hnsw_index_name: Index name for the HNSW index.
        :param hnsw_ef_search: The `ef_search` parameter to use at query time. Only used if search_strategy is set to
            `"hnsw"`. See the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
        :param keyword_index_name: Index name for the keyword GIN index.
        """
        # Initialize connection attributes first so __del__ is safe even if __init__ raises
        self._connection: Connection | None = None
        self._cursor: Cursor | None = None
        self._dict_cursor: Cursor[DictRow] | None = None
        self._connector: Any = None  # google.cloud.alloydbconnector.Connector
        self._table_initialized = False

        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)

        self.instance_uri = instance_uri
        self.user = user
        self.password = password
        self.db = db
        self.enable_iam_auth = enable_iam_auth
        self.ip_type = ip_type
        self.create_extension = create_extension
        self.schema_name = schema_name
        self.table_name = table_name
        self.language = language
        self.embedding_dimension = embedding_dimension
        self.vector_function = vector_function
        self.recreate_table = recreate_table
        self.search_strategy = search_strategy
        self.hnsw_recreate_index_if_exists = hnsw_recreate_index_if_exists
        self.hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}
        self.hnsw_index_name = hnsw_index_name
        self.hnsw_ef_search = hnsw_ef_search
        self.keyword_index_name = keyword_index_name

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            instance_uri=self.instance_uri.to_dict(),
            user=self.user.to_dict(),
            password=self.password.to_dict(),
            db=self.db,
            enable_iam_auth=self.enable_iam_auth,
            ip_type=self.ip_type,
            create_extension=self.create_extension,
            schema_name=self.schema_name,
            table_name=self.table_name,
            language=self.language,
            embedding_dimension=self.embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
            search_strategy=self.search_strategy,
            hnsw_recreate_index_if_exists=self.hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=self.hnsw_index_creation_kwargs,
            hnsw_index_name=self.hnsw_index_name,
            hnsw_ef_search=self.hnsw_ef_search,
            keyword_index_name=self.keyword_index_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlloyDBDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["instance_uri", "user", "password"])
        return default_from_dict(cls, data)

    def close(self) -> None:
        """
        Closes the database connection and the AlloyDB connector.

        Call this when you are done using the document store to release resources.
        For long-lived applications the connector runs a background refresh thread;
        calling `close()` ensures that thread is stopped cleanly.
        """
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: S110
                pass
            self._connection = None
            self._cursor = None
            self._dict_cursor = None

        if self._connector is not None:
            try:
                self._connector.close()
            except Exception:  # noqa: S110
                pass
            self._connector = None

        self._table_initialized = False

    def __del__(self) -> None:
        """
        Closes the connector when the object is garbage collected.
        """
        self.close()

    @staticmethod
    def _connection_is_valid(connection: Connection) -> bool:
        """
        Internal method to check if the connection is still valid.
        """
        try:
            connection.execute("")
        except Error:
            return False
        return True

    @overload
    def _execute_sql(
        self, cursor: Cursor, sql_query: SQL | Composed, params: tuple | None = None, error_msg: str = ""
    ) -> Cursor: ...

    @overload
    def _execute_sql(
        self, cursor: Cursor[DictRow], sql_query: SQL | Composed, params: tuple | None = None, error_msg: str = ""
    ) -> Cursor[DictRow]: ...

    def _execute_sql(
        self,
        cursor: Cursor | Cursor[DictRow],
        sql_query: SQL | Composed,
        params: tuple | None = None,
        error_msg: str = "",
    ) -> Cursor | Cursor[DictRow]:
        """
        Internal method to execute SQL statements and handle exceptions.

        :param cursor: The cursor to use to execute the SQL query.
        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
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

    def _ensure_db_setup(self) -> None:
        """
        Internal method to set up the database connection and initialize the table.

        Establishes a new connection if one does not exist or the existing connection
        is no longer valid. Uses the AlloyDB Python Connector to securely connect.
        """
        if (
            self._connection is not None
            and self._cursor is not None
            and self._dict_cursor is not None
            and self._connection_is_valid(self._connection)
        ):
            return

        # Close stale connection if it exists
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: S110
                pass

        # Lazy-init the connector (one per store instance, reused across calls)
        if self._connector is None:
            try:
                from google.cloud.alloydbconnector import Connector  # type: ignore[import]  # noqa: PLC0415
            except ImportError as exc:
                msg = (
                    "Could not import google-cloud-alloydb-connector. "
                    "Run: pip install 'google-cloud-alloydb-connector[psycopg]'"
                )
                raise ImportError(msg) from exc
            self._connector = Connector(ip_type=self.ip_type)

        instance_uri = self.instance_uri.resolve_value() or ""
        user = self.user.resolve_value() or ""
        password = self.password.resolve_value() if self.password else None

        connect_kwargs: dict[str, Any] = {
            "driver": "psycopg",
            "user": user,
            "db": self.db,
            "enable_iam_auth": self.enable_iam_auth,
        }
        if not self.enable_iam_auth and password:
            connect_kwargs["password"] = password

        try:
            connection: Connection = self._connector.connect(instance_uri, **connect_kwargs)
        except Exception as e:
            msg = (
                "Failed to connect to AlloyDB instance. "
                "Ensure the instance URI is correct, the AlloyDB Connector is authorized, "
                "and the database credentials are valid."
            )
            raise DocumentStoreError(msg) from e

        connection.autocommit = True

        if self.create_extension:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")

        register_vector(connection)

        self._connection = connection
        self._cursor = connection.cursor()
        self._dict_cursor = connection.cursor(row_factory=dict_row)

        if not self._table_initialized:
            self._initialize_table()

    def _build_table_creation_queries(self) -> tuple[SQL, Composed, SQL, Composed]:
        """
        Internal method to build the SQL queries for table creation.
        """
        sql_table_exists = SQL("SELECT 1 FROM pg_tables WHERE schemaname = %s AND tablename = %s")
        sql_create_table = SQL(CREATE_TABLE_STATEMENT).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            embedding_dimension=SQLLiteral(self.embedding_dimension),
            embedding_col_type=SQL("vector"),
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

    def _initialize_table(self) -> None:
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

    def delete_table(self) -> None:
        """
        Deletes the table used to store Haystack documents.

        The name of the schema (`schema_name`) and the name of the table (`table_name`)
        are defined when initializing the `AlloyDBDocumentStore`.
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
            error_msg=f"Could not delete table {self.schema_name}.{self.table_name} in AlloyDBDocumentStore",
        )

    def _build_hnsw_queries(self) -> tuple[Composed | None, SQL, Composed, Composed]:
        """
        Common method to build all HNSW-related SQL queries.
        """
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

        if self.vector_function not in VECTOR_FUNCTION_TO_POSTGRESQL_OPS:
            msg = f"Unsupported vector_function '{self.vector_function}' for HNSW index."
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

    def _handle_hnsw(self) -> None:
        """
        Internal method to handle the HNSW index creation.

        It also sets the `hnsw.ef_search` parameter for queries if it is specified.
        """
        sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index = (
            self._build_hnsw_queries()
        )

        assert self._cursor is not None

        if sql_set_hnsw_ef_search is not None:
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

    def count_documents(self) -> int:
        """
        Returns how many documents are in the document store.

        :returns: The number of documents in the document store.
        """
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        self._ensure_db_setup()
        assert self._cursor is not None

        result = self._execute_sql(
            cursor=self._cursor,
            sql_query=sql_count,
            error_msg="Could not count documents in AlloyDBDocumentStore",
        ).fetchone()

        if result is not None:
            return result[0]
        return 0

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        **Filter operator support**: comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`,
        `not in`, `like`, `not like`) and logical operators `AND` and `OR` are fully supported.
        The `NOT` logical operator is **not** supported — use `!=` or `not in` comparison
        operators instead.

        :param filters: The filters to apply to the document list.

        :raises TypeError: If `filters` is not a dictionary.
        :raises ValueError: If `filters` syntax is invalid.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)

        sql_filter = SQL("SELECT * FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        params: tuple = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_filter += sql_where_clause

        self._ensure_db_setup()
        assert self._dict_cursor is not None

        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_filter,
            params=params,
            error_msg="Could not filter documents from AlloyDBDocumentStore.",
        )

        records = result.fetchall()
        return _from_pg_to_haystack_documents(records)

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

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
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

        db_documents = _from_haystack_to_pg_documents(documents)
        sql_insert = self._build_insert_statement(policy)

        self._ensure_db_setup()
        assert self._cursor is not None
        assert self._connection is not None

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
                f"Could not write documents to AlloyDBDocumentStore. Error: {e!r}. \n"
                "You can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(error_msg) from e

        # Count inserted documents via the RETURNING id clause
        written_docs = 0
        while True:
            if self._cursor.fetchone():
                written_docs += 1
            if not self._cursor.nextset():
                break

        return written_docs

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name} WHERE id = ANY(%s)").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        self._ensure_db_setup()
        assert self._cursor is not None
        self._execute_sql(
            cursor=self._cursor,
            sql_query=delete_sql,
            params=(document_ids,),
            error_msg="Could not delete documents from AlloyDBDocumentStore",
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
            cursor=self._cursor,
            sql_query=query,
            error_msg="Could not delete all documents from AlloyDBDocumentStore",
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

        params: tuple = ()
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
                error_msg="Could not delete documents by filter from AlloyDBDocumentStore",
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
            msg = f"Failed to delete documents by filter from AlloyDBDocumentStore: {e!s}"
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
                error_msg="Could not update documents by filter from AlloyDBDocumentStore",
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
            msg = f"Failed to update documents by filter in AlloyDBDocumentStore: {e!s}"
            raise DocumentStoreError(msg) from e

    def _prepare_filters_count_documents(self, filters: dict[str, Any] | None) -> tuple[tuple[Any, ...], Composed]:
        _validate_filters(filters)
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )
        params: tuple = ()
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
            error_msg="Could not count documents by filter in AlloyDBDocumentStore",
        ).fetchone()

        if result is not None:
            return result[0]
        return 0

    @staticmethod
    def _normalize_metadata_field_name(field_name: str) -> str:
        """
        Normalizes metadata field names by removing the 'meta.' prefix if present.

        :param field_name: The field name to normalize.
        :returns: The normalized field name.
        """
        if field_name.startswith("meta."):
            field_name = field_name[5:]

        # Only allow alphanumeric characters, underscores, and hyphens to prevent SQL injection
        if not all(c.isalnum() or c in ("_", "-") for c in field_name):
            msg = (
                f"Invalid metadata field name: '{field_name}'. Field names can only contain alphanumeric "
                f"characters, underscores, and hyphens."
            )
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
        count_expressions = []
        for field in normalized_fields:
            count_expressions.append(
                SQL("COUNT(DISTINCT meta->>{} ) AS {}").format(SQLLiteral(field), Identifier(field))
            )

        sql_select = SQL("SELECT ") + SQL(", ").join(count_expressions)
        sql_from = SQL(" FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        sql_query = sql_select + sql_from

        params: tuple = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)
            sql_query += sql_where_clause

        return sql_query, params

    @staticmethod
    def _process_count_unique_metadata_result(
        result: dict[str, Any] | None, normalized_fields: list[str]
    ) -> dict[str, int]:
        """
        Processes the result from counting unique metadata values.

        :param result: The database result row, or None if no results.
        :param normalized_fields: List of normalized metadata field names.
        :returns: A dictionary mapping field names to their unique value counts.
        """
        if result is None:
            return dict.fromkeys(normalized_fields, 0)
        return {field: result.get(field, 0) for field in normalized_fields}

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Returns the count of unique values for each specified metadata field.

        Considers only documents that match the provided filters.

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

        normalized_fields = [AlloyDBDocumentStore._normalize_metadata_field_name(f) for f in metadata_fields]
        sql_query, params = self._build_count_unique_metadata_query(normalized_fields, filters)

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg="Could not count unique metadata values in AlloyDBDocumentStore",
        ).fetchone()

        return AlloyDBDocumentStore._process_count_unique_metadata_result(result, normalized_fields)

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
        return "text"

    @staticmethod
    def _analyze_metadata_fields_from_records(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        """
        Analyzes metadata fields from database records and infers their types.

        :param records: List of database records containing 'meta' field.
        :returns: A dictionary mapping field names to their type information.
        """
        fields_info: dict[str, dict[str, str]] = {}

        for record in records:
            meta = record.get("meta")
            if not isinstance(meta, dict):
                continue

            for field_name, field_value in meta.items():
                if field_name not in fields_info:
                    if field_value is not None:
                        inferred_type = AlloyDBDocumentStore._infer_metadata_field_type(field_value)
                        fields_info[field_name] = {"type": inferred_type}
                    else:
                        fields_info[field_name] = {"type": "text"}

        return fields_info

    def _analyze_metadata_from_docs_query(self) -> Composed:
        return SQL("SELECT meta FROM {schema_name}.{table_name} WHERE meta IS NOT NULL").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns information about the metadata fields in the document store.

        Since metadata is stored in a JSONB field, this method analyzes actual data
        to infer field types.

        Example return:
        ```python
        {
            'category': {'type': 'text'},
            'priority': {'type': 'integer'},
        }
        ```

        :returns: A dictionary mapping field names to their type information.
        """
        self._ensure_db_setup()
        assert self._dict_cursor is not None

        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=self._analyze_metadata_from_docs_query(),
            error_msg="Could not retrieve metadata fields info from AlloyDBDocumentStore",
        )

        records = result.fetchall()
        return AlloyDBDocumentStore._analyze_metadata_fields_from_records(records)

    def _build_keyword_retrieval_query(
        self, top_k: int, filters: dict[str, Any] | None = None
    ) -> tuple[Composed, tuple]:
        """
        Builds the SQL query and the where parameters for keyword retrieval.
        """
        sql_select = SQL(KEYWORD_QUERY).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
        )

        where_params: tuple = ()
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
        `AlloyDBDocumentStore` and it should not be called directly.
        `AlloyDBKeywordRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that are most similar to `query`
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        sql_query, where_params = self._build_keyword_retrieval_query(top_k=top_k, filters=filters)

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=(query, *where_params),
            error_msg="Could not retrieve documents from AlloyDBDocumentStore.",
        )

        records = result.fetchall()
        return _from_pg_to_haystack_documents(records)

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
                f"query_embedding dimension ({len(query_embedding)}) does not match AlloyDBDocumentStore "
                f"embedding dimension ({self.embedding_dimension})."
            )
            raise ValueError(msg)

        vector_function = vector_function or self.vector_function
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)

        # The vector must be a string with this format: "'[3,1,2]'"
        query_embedding_for_postgres = f"'[{','.join(str(el) for el in query_embedding)}]'"

        # Compute scores using the approach described in pgvector README:
        # https://github.com/pgvector/pgvector?tab=readme-ov-file#distances
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
        params: tuple = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters)

        # When using l2_distance, sort ASC (smallest distance = most similar)
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
        `AlloyDBDocumentStore` and it should not be called directly.
        `AlloyDBEmbeddingRetriever` uses this method directly and is the public interface for it.

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
            error_msg="Could not retrieve documents from AlloyDBDocumentStore.",
        )

        records = result.fetchall()
        return _from_pg_to_haystack_documents(records)

    def _build_min_max_query(self, normalized_field: str, field_type: str) -> Composed:
        """
        Builds the SQL query for getting min/max values based on the field type.

        :param normalized_field: The normalized metadata field name.
        :param field_type: The type of the field (integer, real, text, or boolean).
        :returns: The SQL query for min/max calculation.
        """
        field_literal = SQLLiteral(normalized_field)

        if field_type == "integer":
            cast = SQL("::integer")
            extractor = SQL("(meta->>{}){}").format(field_literal, cast)
        elif field_type == "real":
            cast = SQL("::real")
            extractor = SQL("(meta->>{}){}").format(field_literal, cast)
        else:
            extractor = SQL('meta->>{} COLLATE "C"').format(field_literal)

        return SQL(
            "SELECT MIN({extractor}) AS min_value, MAX({extractor}) AS max_value "
            "FROM {schema_name}.{table_name} WHERE meta->>{field} IS NOT NULL"
        ).format(
            extractor=extractor,
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            field=field_literal,
        )

    def get_metadata_field_min_max(self, field: str) -> dict[str, Any]:
        """
        Returns the minimum and maximum values for a metadata field.

        For numeric fields (integer, real), returns numeric min/max.
        For text and other non-numeric fields, returns lexicographic min/max
        using the `"C"` collation.

        :param field: The metadata field name (with or without the "meta." prefix).
        :returns: A dictionary with `min` and `max` keys. Returns
            `{"min": None, "max": None}` when the field has no values or the
            store is empty.
        """
        normalized_field = self._normalize_metadata_field_name(field)

        fields_info = self.get_metadata_fields_info()
        if normalized_field not in fields_info:
            return {"min": None, "max": None}

        field_type = fields_info[normalized_field]["type"]
        sql_query = self._build_min_max_query(normalized_field, field_type)

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            error_msg=f"Could not get min/max for field '{field}' from AlloyDBDocumentStore",
        ).fetchone()

        if result is None:
            return {"min": None, "max": None}
        return {"min": result.get("min_value"), "max": result.get("max_value")}

    def get_metadata_field_unique_values(self, field: str, filters: dict[str, Any] | None = None) -> list[Any]:
        """
        Returns a list of unique values for a metadata field.

        :param field: The metadata field name (with or without the "meta." prefix).
        :param filters: Optional filters to restrict the documents considered.
        :returns: A list of unique values for the given field.
        """
        normalized_field = self._normalize_metadata_field_name(field)
        field_literal = SQLLiteral(normalized_field)

        sql_query = SQL(
            "SELECT DISTINCT meta->>{field} AS value FROM {schema_name}.{table_name} WHERE meta->>{field} IS NOT NULL"
        ).format(
            field=field_literal,
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        params: tuple = ()
        if filters:
            _validate_filters(filters)
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(filters, operator="AND")
            sql_query += sql_where_clause

        self._ensure_db_setup()
        assert self._dict_cursor is not None
        result = self._execute_sql(
            cursor=self._dict_cursor,
            sql_query=sql_query,
            params=params,
            error_msg=f"Could not get unique values for field '{field}' from AlloyDBDocumentStore",
        )

        records = result.fetchall()
        return [r["value"] for r in records]
