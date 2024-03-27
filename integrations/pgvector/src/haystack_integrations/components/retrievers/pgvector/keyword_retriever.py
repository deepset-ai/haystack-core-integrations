# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@component
class PgvectorKeywordRetriever:
    """
    Retrieves documents from the `PgvectorDocumentStore`, based on their sparse vectors.

    Example usage:
    ```python
    from haystack.document_stores import DuplicatePolicy
    from haystack import Document, Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

    from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
    from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever

    # Set an environment variable `PG_CONN_STR` with the connection string to your PostgreSQL database.
    # e.g., "postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"

    document_store = PgvectorDocumentStore(
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
    query_pipeline.add_component("retriever", PgvectorKeywordRetriever(document_store=document_store))
    query_pipeline.connect("query", "retriever.query")

    query = "How many languages are there?"

    res = query_pipeline.run({"retriever": {"text": query}})

    assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
    """

    def __init__(
    self,
    *,
    document_store: PgvectorDocumentStore,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    ):
        """
        :param document_store: An instance of `PgvectorDocumentStore}.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.

        :raises ValueError: If `document_store` is not an instance of `PgvectorDocumentStore` or if `vector_function`
            is not one of the valid options.
        """
        if not isinstance(document_store, PgvectorDocumentStore):
            msg = "document_store must be an instance of PgvectorDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PgvectorKeywordRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = PgvectorDocumentStore.from_dict(doc_store_params)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        user_query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Retrieve documents from the `PgvectorDocumentStore`, based on their embeddings.

        :param user_input: The user's query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.

        :returns: List of Documents similar to `user_query`.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        docs = self.document_store._keyword_retrieval(
            user_query=user_query,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
