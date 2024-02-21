# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils import deserialize_secrets_inplace
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


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
    ):
        """
        Create the MongoDBAtlasDocumentStore component.

        :param document_store: An instance of MongoDBAtlasDocumentStore.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        """
        if not isinstance(document_store, MongoDBAtlasDocumentStore):
            msg = "document_store must be an instance of MongoDBAtlasDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasEmbeddingRetriever":
        deserialize_secrets_inplace(
            data["init_parameters"]["document_store"]["init_parameters"], keys=["mongo_connection_string"]
        )
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
    ):
        """
        Retrieve documents from the MongoDBAtlasDocumentStore, based on their embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :return: List of Documents similar to `query_embedding`.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        docs = self.document_store.embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
