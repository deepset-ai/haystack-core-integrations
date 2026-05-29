# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

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
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create the `AlloyDBKeywordRetriever` component.

        :param document_store: An instance of `AlloyDBDocumentStore` to use as the document store.
        :param filters: Filters applied to the retrieved documents.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: Policy to determine how filters are applied at query time.
            `FilterPolicy.REPLACE` (default) replaces the init filters with the run-time filters.
            `FilterPolicy.MERGE` merges the init filters with the run-time filters.
        :raises ValueError: If `document_store` is not an instance of `AlloyDBDocumentStore`.
        """
        if not isinstance(document_store, AlloyDBDocumentStore):
            msg = "document_store must be an instance of AlloyDBDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

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
        :param filters: Filters applied to the retrieved documents.
            The `filter_policy` set at initialization determines how these are combined with the init filters.
        :param top_k: Maximum number of documents to return. Overrides the `top_k` set at initialization.
        :returns: A dictionary containing the `documents` retrieved from the document store.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store._keyword_retrieval(
            query=query,
            filters=filters,
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
            filter_policy=self.filter_policy.value,
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
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
