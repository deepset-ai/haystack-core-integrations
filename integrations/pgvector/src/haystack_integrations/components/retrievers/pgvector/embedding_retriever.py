# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Literal, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.document_stores.pgvector.document_store import VALID_VECTOR_FUNCTIONS


@component
class PgvectorEmbeddingRetriever:
    """
    Retrieves documents from the PgvectorDocumentStore, based on their dense embeddings.

    Needs to be connected to the PgvectorDocumentStore.
    """

    def __init__(
        self,
        *,
        document_store: PgvectorDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]] = None,
    ):
        """
        Create the PgvectorEmbeddingRetriever component.

        :param document_store: An instance of PgvectorDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param top_k: Maximum number of Documents to return, defaults to 10.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            Defaults to the one set in the `document_store` instance.
            "cosine_similarity" and "inner_product" are similarity functions and
            higher scores indicate greater similarity between the documents.
            "l2_distance" returns the straight-line distance between vectors,
            and the most similar documents are the ones with the smallest score.

            Important: if the document store is using the "hnsw" search strategy, the vector function
            should match the one utilized during index creation to take advantage of the index.
        :type vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"]

        :raises ValueError: If `document_store` is not an instance of PgvectorDocumentStore.
        """
        if not isinstance(document_store, PgvectorDocumentStore):
            msg = "document_store must be an instance of PgvectorDocumentStore"
            raise ValueError(msg)

        if vector_function and vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.vector_function = vector_function or document_store.vector_function

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            vector_function=self.vector_function,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PgvectorEmbeddingRetriever":
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = PgvectorDocumentStore.from_dict(doc_store_params)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]] = None,
    ):
        """
        Retrieve documents from the PgvectorDocumentStore, based on their embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param vector_function: The similarity function to use when searching for similar embeddings.
        :type vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"]
        :return: List of Documents similar to `query_embedding`.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k
        vector_function = vector_function or self.vector_function

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            vector_function=vector_function,
        )
        return {"documents": docs}
