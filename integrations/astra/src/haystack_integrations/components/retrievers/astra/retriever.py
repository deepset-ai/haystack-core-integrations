# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict

from haystack_integrations.document_stores.astra import AstraDocumentStore


@component
class AstraEmbeddingRetriever:
    """
    A component for retrieving documents from an AstraDocumentStore.

    Usage example:
    ```python
    from haystack_integrations.document_stores.astra import AstraDocumentStore
    from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever

    document_store = AstraDocumentStore(
        api_endpoint=api_endpoint,
        token=token,
        collection_name=collection_name,
        duplicates_policy=DuplicatePolicy.SKIP,
        embedding_dim=384,
    )

    retriever = AstraEmbeddingRetriever(document_store=document_store)
    ```
    """

    def __init__(self, document_store: AstraDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        :param filters: a dictionary with filters to narrow down the search space.
        :param top_k: the maximum number of documents to retrieve.
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

        if not isinstance(document_store, AstraDocumentStore):
            message = "document_store must be an instance of AstraDocumentStore"
            raise Exception(message)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """Retrieve documents from the AstraDocumentStore.

        :param query_embedding: floats representing the query embedding
        :param filters: filters to narrow down the search space.
        :param top_k: the maximum number of documents to retrieve.
        :returns: a dictionary with the following keys:
            - `documents`: A list of documents retrieved from the AstraDocumentStore.
        """

        if not top_k:
            top_k = self.top_k

        if not filters:
            filters = self.filters

        return {"documents": self.document_store.search(query_embedding, top_k, filters=filters)}

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
    def from_dict(cls, data: Dict[str, Any]) -> "AstraEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        document_store = AstraDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)
