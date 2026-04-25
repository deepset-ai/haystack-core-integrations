# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore


@component
class AlloyDBKeywordRetriever:
    """
    Retrieves documents from the `AlloyDBDocumentStore` by keyword search.

    Uses PostgreSQL full-text search (`to_tsvector` / `plainto_tsquery`) to find documents.
    Must be connected to the `AlloyDBDocumentStore`.
    """

    def __init__(
        self,
        *,
        document_store: AlloyDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> None:
        """
        Create the `AlloyDBKeywordRetriever` component.

        :param document_store: An instance of `AlloyDBDocumentStore` to use as the document store.
        :param filters: Filters applied to the retrieved documents.
        :param top_k: Maximum number of documents to return.
        :raises ValueError: If `document_store` is not an instance of `AlloyDBDocumentStore`.
        """
        if not isinstance(document_store, AlloyDBDocumentStore):
            msg = "document_store must be an instance of AlloyDBDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the `AlloyDBDocumentStore` by keyword search.

        :param query: A keyword query to search for.
        :param filters: Filters applied to the retrieved documents. Overrides the filters set at initialization.
        :param top_k: Maximum number of documents to return. Overrides the `top_k` set at initialization.
        :returns: A dictionary containing the `documents` retrieved from the document store.
        """
        docs = self.document_store._keyword_retrieval(
            query=query,
            filters=filters or self.filters,
            top_k=top_k or self.top_k,
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
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlloyDBKeywordRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        document_store = AlloyDBDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)
