# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore


@component
class AzureDocumentDBFullTextRetriever:
    """Retrieve documents using Azure DocumentDB BM25 full-text search, currently a gated preview."""

    def __init__(
        self,
        *,
        document_store: AzureDocumentDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create the full-text retriever.

        :param document_store: Azure DocumentDB document store to query.
        :param filters: Default Haystack metadata filters.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: Policy for combining initialization and runtime filters.
        """
        if not isinstance(document_store, AzureDocumentDBDocumentStore):
            msg = "document_store must be an instance of AzureDocumentDBDocumentStore"
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero"
            raise ValueError(msg)
        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AzureDocumentDBFullTextRetriever":
        """Deserialize this component from a dictionary."""
        data["init_parameters"]["document_store"] = AzureDocumentDBDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        if policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(policy)
        return default_from_dict(cls, data)

    def close(self) -> None:
        """Release synchronous document store resources."""
        self.document_store.close()

    async def close_async(self) -> None:
        """Release asynchronous document store resources."""
        await self.document_store.close_async()

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str | list[str],
        fuzzy: dict[str, int] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents by BM25 keyword search.

        :param query: Query string or strings.
        :param fuzzy: Azure DocumentDB fuzzy-search options such as `maxEdits`.
        :param filters: Runtime Haystack metadata filters.
        :param top_k: Runtime maximum number of documents.
        :returns: A dictionary containing the retrieved `documents`.
        """
        effective_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        documents = self.document_store._full_text_retrieval(
            query=query,
            fuzzy=fuzzy,
            filters=effective_filters,
            top_k=self.top_k if top_k is None else top_k,
        )
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query: str | list[str],
        fuzzy: dict[str, int] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents by BM25 keyword search.

        :param query: Query string or strings.
        :param fuzzy: Azure DocumentDB fuzzy-search options such as `maxEdits`.
        :param filters: Runtime Haystack metadata filters.
        :param top_k: Runtime maximum number of documents.
        :returns: A dictionary containing the retrieved `documents`.
        """
        effective_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        documents = await self.document_store._full_text_retrieval_async(
            query=query,
            fuzzy=fuzzy,
            filters=effective_filters,
            top_k=self.top_k if top_k is None else top_k,
        )
        return {"documents": documents}
