# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Literal, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.document_stores.mongodb_atlas.document_store import METRIC_TYPES


@component
class MongoDBAtlasEmbeddingRetriever:
    """
    Retrieves documents from the MongoDBAtlasDocumentStore by embedding similarity.

    Needs to be connected to the MongoDBAtlasDocumentStore.
    """

    def __init__(
        self,
        *,
        document_store: MongoDBAtlasDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        similarity: Literal["euclidean", "cosine", "dotProduct"] = "cosine",
    ):
        """
        Create the MongoDBAtlasDocumentStore component.

        :param document_store: An instance of MongoDBAtlasDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param top_k: Maximum number of Documents to return, defaults to 10.
        :param similarity: The similarity function to use when searching for similar embeddings. 
            Defaults to "cosine". Valid values are "cosine", "euclidean", "dotProduct".
        """
        if not isinstance(document_store, MongoDBAtlasDocumentStore):
            msg = "document_store must be an instance of MongoDBAtlasDocumentStore"
            raise ValueError(msg)

        if similarity not in METRIC_TYPES:
            msg = f"vector_function must be one of {METRIC_TYPES}"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.similarity = similarity

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            similarity=self.similarity,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasEmbeddingRetriever":
        data["init_parameters"]["document_store"] = default_from_dict(
            MongoDBAtlasDocumentStore, data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        similarity: Optional[Literal["euclidean", "cosine", "dotProduct"]] = None,
    ):
        """
        Retrieve documents from the MongoDBAtlasDocumentStore, based on their embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param similarity: The similarity function to use when searching for similar embeddings.
            Defaults to the value provided in the constructor. Valid values are "cosine", "euclidean", "dotProduct".
        :return: List of Documents similar to `query_embedding`.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k
        similarity = similarity or self.similarity

        docs = self.document_store.embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            similarity=similarity,
        )
        return {"documents": docs}
