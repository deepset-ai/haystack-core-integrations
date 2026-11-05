# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


@component
class QdrantMultiVectorRetriever:
    """Search stored multivectors using dot-product MaxSim in Qdrant."""

    def __init__(
        self,
        document_store: QdrantDocumentStore,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        return_embedding: bool = False,
    ) -> None:
        if not isinstance(document_store, QdrantDocumentStore):
            msg = "document_store must be a QdrantDocumentStore."
            raise TypeError(msg)
        if document_store.multi_vector_embedding_dim is None:
            msg = "The Document Store must have multi_vector_embedding_dim configured."
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero."
            raise ValueError(msg)
        self.document_store = document_store
        self.top_k = top_k
        self.filters = filters
        self.filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy
        self.return_embedding = return_embedding

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retriever and its Document Store configuration."""
        return default_to_dict(
            self,
            document_store=self.document_store,
            top_k=self.top_k,
            filters=self.filters,
            filter_policy=self.filter_policy.value,
            return_embedding=self.return_embedding,
        )

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        return_embedding: bool | None = None,
    ) -> dict[str, list[Document]]:
        """Return the best matches, optionally including their stored embeddings."""
        filters = apply_filter_policy(
            filter_policy=self.filter_policy, init_filters=self.filters, runtime_filters=filters
        )
        documents = self.document_store._query_by_multi_vector_embedding(
            query_embedding=query_embedding,
            filters=filters,
            top_k=self.top_k if top_k is None else top_k,
            return_embedding=self.return_embedding if return_embedding is None else return_embedding,
        )
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        return_embedding: bool | None = None,
    ) -> dict[str, list[Document]]:
        """Asynchronously search using Qdrant's async client."""
        filters = apply_filter_policy(
            filter_policy=self.filter_policy, init_filters=self.filters, runtime_filters=filters
        )
        documents = await self.document_store._query_by_multi_vector_embedding_async(
            query_embedding=query_embedding,
            filters=filters,
            top_k=self.top_k if top_k is None else top_k,
            return_embedding=self.return_embedding if return_embedding is None else return_embedding,
        )
        return {"documents": documents}

    def close(self) -> None:
        """Close the synchronous Document Store client."""
        self.document_store.close()

    async def close_async(self) -> None:
        """Close the asynchronous Document Store client."""
        await self.document_store.close_async()
