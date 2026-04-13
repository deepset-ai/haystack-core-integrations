from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.oracle import OracleDocumentStore


def _merge_filters(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """AND-merge two Haystack filter dicts. Returns None if both are empty."""
    base = base or {}
    override = override or {}
    if not base and not override:
        return None
    if not base:
        return override
    if not override:
        return base
    return {"operator": "AND", "conditions": [base, override]}


@component
class OracleEmbeddingRetriever:
    """Retrieves documents from an OracleDocumentStore using vector similarity.

    Use inside a Haystack pipeline after a text embedder::

        pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        pipeline.add_component("retriever", OracleEmbeddingRetriever(
            document_store=store, top_k=5
        ))
        pipeline.connect("embedder.embedding", "retriever.query_embedding")
    """

    def __init__(
        self,
        *,
        document_store: OracleDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> None:
        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """Retrieve documents by vector similarity.

        Args:
            query_embedding: Dense float vector from an embedder component.
            filters: Runtime filters, AND-merged with constructor filters.
            top_k: Override the constructor top_k for this call.

        Returns:
            ``{"documents": [Document, ...]}``
        """
        merged = _merge_filters(self.filters, filters)
        docs = self.document_store._embedding_retrieval(
            query_embedding,
            filters=merged,
            top_k=top_k if top_k is not None else self.top_k,
        )
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """Async variant of :meth:`run`."""
        merged = _merge_filters(self.filters, filters)
        docs = await self.document_store._embedding_retrieval_async(
            query_embedding,
            filters=merged,
            top_k=top_k if top_k is not None else self.top_k,
        )
        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleEmbeddingRetriever":
        params = data.get("init_parameters", {})
        if "document_store" in params:
            params["document_store"] = OracleDocumentStore.from_dict(params["document_store"])
        return default_from_dict(cls, data)
