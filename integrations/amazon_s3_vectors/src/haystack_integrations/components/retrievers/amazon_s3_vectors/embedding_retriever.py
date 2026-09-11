# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


@component
class S3VectorsEmbeddingRetriever:
    """
    Retrieve documents from an `S3VectorsDocumentStore` based on their dense embeddings.

    Usage example:
    ```python
    from haystack import Document, Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack.document_stores.types import DuplicatePolicy
    from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
    from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore

    document_store = S3VectorsDocumentStore(
        vector_bucket_name="my-vectors",
        index_name="my-index",
        dimension=768,
    )

    documents = [
        Document(content="There are over 7,000 languages spoken around the world today."),
        Document(content="Elephants have been observed to behave in a way that indicates..."),
        Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
    ]

    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", S3VectorsEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "How many languages are there?"
    res = query_pipeline.run({"text_embedder": {"text": query}})
    ```
    """

    def __init__(
        self,
        *,
        document_store: S3VectorsDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the S3VectorsEmbeddingRetriever.

        :param document_store: An instance of `S3VectorsDocumentStore`.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an `S3VectorsDocumentStore`.
        """
        if not isinstance(document_store, S3VectorsDocumentStore):
            msg = "document_store must be an instance of S3VectorsDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "S3VectorsEmbeddingRetriever":
        """Deserialize the component from a dictionary."""
        data["init_parameters"]["document_store"] = S3VectorsDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the S3VectorsDocumentStore based on dense embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
            the `filter_policy` chosen at retriever initialization. Filters are applied server-side during
            the vector search.
        :param top_k: Maximum number of Documents to return. S3 Vectors caps this at 100.
        :returns: A dictionary with key `"documents"` containing the retrieved Documents.
            Returned documents will not contain embeddings.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
