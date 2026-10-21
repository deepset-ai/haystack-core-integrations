# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import suppress
from typing import Any, Literal

from haystack import default_from_dict, default_to_dict
from haystack.document_stores.errors import DocumentStoreError
from haystack.utils.auth import Secret
from psycopg import AsyncConnection, Connection

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


class AlloyDBDocumentStore(PgvectorDocumentStore):
    """
    A Document Store backed by [Google Cloud AlloyDB](https://cloud.google.com/alloydb).

    Uses the [pgvector extension](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings) for vector search.

    AlloyDB is a fully managed, PostgreSQL-compatible database service on Google Cloud.
    Connection is handled securely via the
    [AlloyDB Python Connector](https://github.com/GoogleCloudPlatform/alloydb-python-connector),
    which provides TLS encryption and IAM-based authorization without requiring manual SSL certificate
    management, firewall rules, or IP allowlisting.

    This store extends `PgvectorDocumentStore` and only changes how the connection is opened, so all
    document, filtering, metadata and retrieval operations behave as in `PgvectorDocumentStore`.

    **Filter limitations**: the `NOT` logical operator is not supported. Use `!=` or `not in`
    comparison operators to express negation.

    **Async**: the async methods are not supported, because the AlloyDB Python Connector does not
    provide async psycopg connections. They raise `NotImplementedError`.

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
        super().__init__(
            create_extension=create_extension,
            schema_name=schema_name,
            table_name=table_name,
            language=language,
            embedding_dimension=embedding_dimension,
            vector_function=vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
            hnsw_recreate_index_if_exists=hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=hnsw_index_creation_kwargs,
            hnsw_index_name=hnsw_index_name,
            hnsw_ef_search=hnsw_ef_search,
            keyword_index_name=keyword_index_name,
        )
        self.instance_uri = instance_uri
        self.user = user
        self.password = password
        self.db = db
        self.enable_iam_auth = enable_iam_auth
        self.ip_type = ip_type
        self._connector: Any = None  # google.cloud.alloydbconnector.Connector, created on first connection

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
        return default_from_dict(cls, data)

    def close(self) -> None:
        """
        Release the associated synchronous resources, including the AlloyDB Python Connector.
        """
        super().close()

        if self._connector is not None:
            with suppress(Exception):
                self._connector.close()
            self._connector = None

        self._table_initialized = False

    def _create_connection(self) -> Connection:
        """
        Opens a new connection to AlloyDB through the AlloyDB Python Connector.

        The connector is created on first use and reused for subsequent connections.

        :raises ImportError: If `google-cloud-alloydb-connector` is not installed.
        :raises DocumentStoreError: If the connection cannot be opened.
        """
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
        return connection

    async def _create_connection_async(self) -> AsyncConnection:
        """
        Not supported: the AlloyDB Python Connector does not provide async psycopg connections.

        :raises NotImplementedError: Always.
        """
        msg = (
            "AlloyDBDocumentStore does not support async operations, because the AlloyDB Python Connector "
            "does not provide async psycopg connections. Use the synchronous methods instead."
        )
        raise NotImplementedError(msg)
