# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import default_from_dict, default_to_dict, logging
from haystack.document_stores.errors import DocumentStoreError
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from psycopg import AsyncConnection, Connection, Error
from psycopg.rows import dict_row

from pgvector.psycopg import register_vector, register_vector_async

from ._base import (
    HNSW_INDEX_CREATION_VALID_KWARGS,
    VALID_VECTOR_FUNCTIONS,
    VECTOR_FUNCTION_TO_POSTGRESQL_OPS,
    PostgreSQLDocumentStore,
)

__all__ = [
    "HNSW_INDEX_CREATION_VALID_KWARGS",
    "VALID_VECTOR_FUNCTIONS",
    "VECTOR_FUNCTION_TO_POSTGRESQL_OPS",
    "PgvectorDocumentStore",
]

logger = logging.getLogger(__name__)


class PgvectorDocumentStore(PostgreSQLDocumentStore):
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
    ) -> None:
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
        super().__init__(
            schema_name=schema_name,
            table_name=table_name,
            language=language,
            embedding_dimension=embedding_dimension,
            vector_type=vector_type,
            vector_function=vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
            hnsw_recreate_index_if_exists=hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=hnsw_index_creation_kwargs,
            hnsw_index_name=hnsw_index_name,
            hnsw_ef_search=hnsw_ef_search,
            keyword_index_name=keyword_index_name,
        )
        self.connection_string = connection_string
        self.create_extension = create_extension

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

    def _ensure_db_setup(self) -> None:
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

    async def _ensure_db_setup_async(self) -> None:
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
