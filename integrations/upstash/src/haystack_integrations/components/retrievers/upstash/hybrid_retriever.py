from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document, SparseEmbedding

from haystack_integrations.document_stores.upstash import UpstashDocumentStore


@component
class UpstashHybridRetriever:
    """
    A component for retrieving documents from an UpstashDocumentStore.

    Uses both dense and sparse embeddings, combined via Upstash Vector's native Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        document_store: UpstashDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> None:
        """
        Initializes the UpstashHybridRetriever.

        :param document_store: The UpstashDocumentStore instance to retrieve documents from.
        :param filters: Optional filters to narrow down the search space.
        :param top_k: The maximum number of documents to retrieve.
        """
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this Retriever to a dictionary.

        :returns: The serialized Retriever.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UpstashHybridRetriever":
        """
        Deserializes a dictionary into a Retriever.

        :param data: The serialized Retriever.
        :returns: The deserialized Retriever.
        """
        data["init_parameters"]["document_store"] = UpstashDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        query_sparse_embedding: SparseEmbedding,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieves documents matching the given dense and sparse query embeddings.

        :param query_embedding: The dense embedding to query for.
        :param query_sparse_embedding: The sparse embedding to query for.
        :param filters: Optional filters to narrow down the search space.
        :param top_k: The maximum number of documents to retrieve.
        :returns: A dictionary with the following keys:
            - `documents`: List of documents matching the query.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        from haystack_integrations.document_stores.upstash.filters import _normalize_filters  # noqa: PLC0415

        filter_str = ""
        if filters:
            filter_str = _normalize_filters(filters)

        # Parse query_sparse_embedding
        sparse_vec = (list(query_sparse_embedding.indices), list(query_sparse_embedding.values))

        results = self.document_store._index.query(
            vector=query_embedding,
            sparse_vector=sparse_vec,
            top_k=top_k,
            filter=filter_str,
            include_metadata=True,
            include_vectors=True,
            include_data=True,
        )

        documents = []
        for res in results:
            documents.append(
                Document(
                    id=res.id,
                    content=res.data,
                    embedding=res.vector,
                    meta=res.metadata or {},
                    score=res.score,
                )
            )

        return {"documents": documents}
