# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


class SupabasePgvectorDocumentStore(PgvectorDocumentStore):
    """
    A Document Store for Supabase, using PostgreSQL with the pgvector extension.

    It should be used with Supabase installed.

    This is a thin wrapper around `PgvectorDocumentStore` with Supabase-specific defaults:
    - Reads the connection string from the `SUPABASE_DB_URL` environment variable.
    - Defaults `create_extension` to `False` since pgvector is pre-installed on Supabase.

    **Connection notes:** Supabase offers two pooler ports — transaction mode (6543) and session mode (5432).
    For best compatibility with pgvector operations, use session mode (port 5432) or a direct connection.

    Example usage:

    # Set an environment variable `SUPABASE_DB_URL` with the connection string to your Supabase database.
    ```bash
    export SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:5432/postgres
    ```

    ```python
    from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore

    document_store = SupabasePgvectorDocumentStore(
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
    )
    ```
    """

    def __init__(
        self,
        *,
        connection_string: Secret = Secret.from_env_var("SUPABASE_DB_URL"),
        create_extension: bool = False,
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
        Creates a new SupabasePgvectorDocumentStore instance.

        :param connection_string: The connection string for the Supabase PostgreSQL database, defined as an
            environment variable. Default: `SUPABASE_DB_URL`. Format:
            `postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres`
        :param create_extension: Whether to create the pgvector extension if it doesn't exist.
            Defaults to `False` since Supabase has pgvector pre-installed.
        :param schema_name: The name of the schema the table is created in.
        :param table_name: The name of the table to use to store Haystack documents.
        :param language: The language to be used to parse query and document content in keyword retrieval.
        :param embedding_dimension: The dimension of the embedding.
        :param vector_type: The type of vector used for embedding storage. `"vector"` or `"halfvec"`.
        :param vector_function: The similarity function to use when searching for similar embeddings.
        :param recreate_table: Whether to recreate the table if it already exists.
        :param search_strategy: The search strategy to use: `"exact_nearest_neighbor"` or `"hnsw"`.
        :param hnsw_recreate_index_if_exists: Whether to recreate the HNSW index if it already exists.
        :param hnsw_index_creation_kwargs: Additional keyword arguments for HNSW index creation.
        :param hnsw_index_name: Index name for the HNSW index.
        :param hnsw_ef_search: The `ef_search` parameter to use at query time for HNSW.
        :param keyword_index_name: Index name for the Keyword index.
        """
        super().__init__(
            connection_string=connection_string,
            create_extension=create_extension,
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
    def from_dict(cls, data: dict[str, Any]) -> "SupabasePgvectorDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["connection_string"])
        return default_from_dict(cls, data)
