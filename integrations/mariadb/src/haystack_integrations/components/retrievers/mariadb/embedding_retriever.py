# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
from haystack_integrations.document_stores.mariadb.document_store import VALID_VECTOR_FUNCTIONS


@component
class MariaDBEmbeddingRetriever:
    """
    Retrieves documents from `MariaDBDocumentStore` using vector similarity search.

    Uses MariaDB's native `VEC_DISTANCE_COSINE` or `VEC_DISTANCE_EUCLIDEAN` functions
    with MHNSW indexing for efficient approximate nearest-neighbour search.

    ### Usage example

    ```python
    from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
    from haystack_integrations.components.retrievers.mariadb import MariaDBEmbeddingRetriever

    store = MariaDBDocumentStore(host="localhost", database="haystack", embedding_dimension=768)
    retriever = MariaDBEmbeddingRetriever(document_store=store, top_k=5)
    result = retriever.run(query_embedding=[0.1] * 768)
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        document_store: MariaDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        score_threshold: float | None = None,
        vector_function: Literal["cosine", "euclidean"] | None = None,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the MariaDBEmbeddingRetriever.

        :param document_store: A `MariaDBDocumentStore` instance.
        :param filters: Default Haystack metadata filters applied to every query.
        :param top_k: Maximum number of documents to return.
        :param score_threshold: Minimum score to include a document. Documents below this score are excluded.
        :param vector_function: Override the store's default vector function for this retriever.
        :param filter_policy: How runtime filters interact with init-time filters.
        :raises ValueError: If `document_store` is not a `MariaDBDocumentStore` or
            `vector_function` is invalid.
        """
        if not isinstance(document_store, MariaDBDocumentStore):
            msg = "document_store must be an instance of MariaDBDocumentStore"
            raise ValueError(msg)
        if vector_function and vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.vector_function = vector_function or document_store.vector_function
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            vector_function=self.vector_function,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MariaDBEmbeddingRetriever":
        """Deserialize the component from a dictionary."""
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = MariaDBDocumentStore.from_dict(doc_store_params)
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        vector_function: Literal["cosine", "euclidean"] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents similar to the query embedding.

        :param query_embedding: The query vector.
        :param filters: Runtime filters merged with init-time filters per `filter_policy`.
        :param top_k: Override the retriever's `top_k`.
        :param score_threshold: Override the retriever's `score_threshold`.
        :param vector_function: Override the vector function for this call.
        :returns: Dictionary with `"documents"` key containing the ranked results.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        return {
            "documents": self.document_store._embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k or self.top_k,
                score_threshold=score_threshold if score_threshold is not None else self.score_threshold,
                vector_function=vector_function or self.vector_function,
            )
        }
