# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore


@component
class AlloyDBEmbeddingRetriever:
    """
    Retrieves documents from the `AlloyDBDocumentStore` by embedding similarity.

    Must be connected to the `AlloyDBDocumentStore`.
    """

    def __init__(
        self,
        *,
        document_store: AlloyDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] | None = None,
    ) -> None:
        """
        Create the `AlloyDBEmbeddingRetriever` component.

        :param document_store: An instance of `AlloyDBDocumentStore` to use as the document store.
        :param filters: Filters applied to the retrieved documents.
        :param top_k: Maximum number of documents to return.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            Overrides the `vector_function` set in the `AlloyDBDocumentStore`.
            `"cosine_similarity"` and `"inner_product"` are similarity functions and
            higher scores indicate greater similarity between the documents.
            `"l2_distance"` returns the straight-line distance between vectors,
            and the most similar documents are the ones with the smallest score.
            **Important**: when using the `"hnsw"` search strategy, make sure to use the same
            vector function as the one used when the HNSW index was created.
            If not specified, the `vector_function` of the `AlloyDBDocumentStore` is used.
        :raises ValueError: If `document_store` is not an instance of `AlloyDBDocumentStore`.
        """
        if not isinstance(document_store, AlloyDBDocumentStore):
            msg = "document_store must be an instance of AlloyDBDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.vector_function = vector_function

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the `AlloyDBDocumentStore` by embedding similarity.

        :param query_embedding: A vector representation of the query.
        :param filters: Filters applied to the retrieved documents. Overrides the filters set at initialization.
        :param top_k: Maximum number of documents to return. Overrides the `top_k` set at initialization.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            Overrides the `vector_function` set at initialization.
        :returns: A dictionary containing the `documents` retrieved from the document store.
        """
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters or self.filters,
            top_k=top_k or self.top_k,
            vector_function=vector_function or self.vector_function,
        )
        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            vector_function=self.vector_function,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlloyDBEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        document_store = AlloyDBDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)
