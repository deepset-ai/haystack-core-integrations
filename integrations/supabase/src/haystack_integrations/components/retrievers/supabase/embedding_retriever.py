# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component, default_from_dict
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector.document_store import VALID_VECTOR_FUNCTIONS
from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore


@component
class SupabasePgvectorEmbeddingRetriever(PgvectorEmbeddingRetriever):
    """
    Retrieves documents from the `SupabasePgvectorDocumentStore`, based on their dense embeddings.

    This is a thin wrapper around `PgvectorEmbeddingRetriever`, adapted for use with
    `SupabasePgvectorDocumentStore`.

    Example usage:
    ```python
    from haystack.document_stores import DuplicatePolicy
    from haystack import Document, Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

    from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore
    from haystack_integrations.components.retrievers.supabase import SupabasePgvectorEmbeddingRetriever

    # Set an environment variable `SUPABASE_DB_URL` with the connection string to your Supabase database.

    document_store = SupabasePgvectorDocumentStore(
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
    )

    documents = [Document(content="There are over 7,000 languages spoken around the world today."),
                 Document(content="Elephants have been observed to behave in a way that indicates..."),
                 Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", SupabasePgvectorEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "How many languages are there?"

    res = query_pipeline.run({"text_embedder": {"text": query}})

    assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
    ```
    """

    def __init__(
        self,
        *,
        document_store: SupabasePgvectorDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] | None = None,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the SupabasePgvectorEmbeddingRetriever.

        :param document_store: An instance of `SupabasePgvectorDocumentStore`.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            Defaults to the one set in the `document_store` instance.
            `"cosine_similarity"` and `"inner_product"` are similarity functions and
            higher scores indicate greater similarity between the documents.
            `"l2_distance"` returns the straight-line distance between vectors,
            and the most similar documents are the ones with the smallest score.
            **Important**: if the document store is using the `"hnsw"` search strategy, the vector function
            should match the one utilized during index creation to take advantage of the index.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of `SupabasePgvectorDocumentStore` or if
            `vector_function` is not one of the valid options.
        """
        if not isinstance(document_store, SupabasePgvectorDocumentStore):
            msg = "document_store must be an instance of SupabasePgvectorDocumentStore"
            raise ValueError(msg)

        if vector_function and vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}"
            raise ValueError(msg)

        # Set attributes directly rather than calling super().__init__() because Haystack's
        # @component metaclass wraps __init__, making super() fail across two @component classes.
        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.vector_function = vector_function or document_store.vector_function
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupabasePgvectorEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = SupabasePgvectorDocumentStore.from_dict(doc_store_params)
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
