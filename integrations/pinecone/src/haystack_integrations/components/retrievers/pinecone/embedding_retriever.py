# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@component
class PineconeEmbeddingRetriever:
    """
    Retrieves documents from the PineconeDocumentStore, based on their dense embeddings.

    Needs to be connected to the PineconeDocumentStore.
    """

    def __init__(
        self,
        *,
        document_store: PineconeDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ):
        """
        Create the PineconeEmbeddingRetriever component.

        :param document_store: An instance of PineconeDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param top_k: Maximum number of Documents to return, defaults to 10.

        :raises ValueError: If `document_store` is not an instance of PineconeDocumentStore.
        """
        if not isinstance(document_store, PineconeDocumentStore):
            msg = "document_store must be an instance of PineconeDocumentStore"
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
    def from_dict(cls, data: Dict[str, Any]) -> "PineconeEmbeddingRetriever":
        data["init_parameters"]["document_store"] = default_from_dict(
            PineconeDocumentStore, data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float]):
        """
        Retrieve documents from the PineconeDocumentStore, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :return: List of Document similar to `query_embedding`.
        """
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=self.filters,
            top_k=self.top_k,
        )
        return {"documents": docs}
