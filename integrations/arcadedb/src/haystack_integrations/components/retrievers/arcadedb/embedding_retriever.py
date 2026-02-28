# SPDX-FileCopyrightText: 2025-present ArcadeData Ltd <info@arcadedb.com>
#
# SPDX-License-Identifier: Apache-2.0

"""ArcadeDB Embedding Retriever for Haystack 2.x pipelines."""

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore


@component
class ArcadeDBEmbeddingRetriever:
    """
    Retrieve documents from ArcadeDB using vector similarity (LSM_VECTOR / HNSW index).

    Usage example:

    ```python
    from haystack_integrations.components.retrievers.arcadedb import ArcadeDBEmbeddingRetriever
    from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

    store = ArcadeDBDocumentStore(database="mydb")
    retriever = ArcadeDBEmbeddingRetriever(document_store=store, top_k=5)
    ```
    """

    def __init__(
        self,
        *,
        document_store: ArcadeDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ):
        """
        Create an ArcadeDBEmbeddingRetriever.

        :param document_store: An instance of ``ArcadeDBDocumentStore``.
        :param filters: Default filters applied to every retrieval call.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: How runtime filters interact with default filters.
        """
        self._document_store = document_store
        self._filters = filters
        self._top_k = top_k
        self._filter_policy = filter_policy

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents by vector similarity.

        :param query_embedding: The embedding vector to search with.
        :param filters: Optional filters to narrow results.
        :param top_k: Maximum number of documents to return.
        :returns: A dict with key ``"documents"`` containing the retrieved documents.
        """
        effective_top_k = top_k if top_k is not None else self._top_k

        if self._filter_policy == FilterPolicy.REPLACE and filters is not None:
            effective_filters = filters
        elif self._filter_policy == FilterPolicy.MERGE and filters is not None and self._filters is not None:
            effective_filters = {
                "operator": "AND",
                "conditions": [self._filters, filters],
            }
        else:
            effective_filters = filters or self._filters

        documents = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=effective_filters,
            top_k=effective_top_k,
        )
        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this retriever to a dictionary."""
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArcadeDBEmbeddingRetriever":
        """Deserialize an ArcadeDBEmbeddingRetriever from a dictionary."""
        init_params = data.get("init_parameters", {})
        if "document_store" in init_params:
            init_params["document_store"] = ArcadeDBDocumentStore.from_dict(init_params["document_store"])
        if "filter_policy" in init_params:
            init_params["filter_policy"] = FilterPolicy(init_params["filter_policy"])
        return default_from_dict(cls, data)
