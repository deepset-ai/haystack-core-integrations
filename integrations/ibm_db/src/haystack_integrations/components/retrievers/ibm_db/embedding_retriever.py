# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.ibm_db import Db2DocumentStore


@component
class Db2EmbeddingRetriever:
    """
    Retrieves documents from a Db2DocumentStore using vector similarity.

    Use inside a Haystack pipeline after a text embedder::

        pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        pipeline.add_component("retriever", Db2EmbeddingRetriever(
            document_store=store, top_k=5
        ))
        pipeline.connect("embedder.embedding", "retriever.query_embedding")
    """

    def __init__(
        self,
        *,
        document_store: Db2DocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        if not isinstance(document_store, Db2DocumentStore):
            msg = "document_store must be an instance of Db2DocumentStore"
            raise TypeError(msg)
        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents by vector similarity.

        Args:
            query_embedding: Dense float vector from an embedder component.
            filters: Runtime filters, merged with constructor filters according to filter_policy.
            top_k: Override the constructor top_k for this call.

        Returns:
            ``{"documents": [Document, ...]}``
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store._embedding_retrieval(
            query_embedding,
            filters=filters,
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
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = await self.document_store._embedding_retrieval_async(
            query_embedding,
            filters=filters,
            top_k=top_k if top_k is not None else self.top_k,
        )
        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Db2EmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        params = data.get("init_parameters", {})
        if "document_store" in params:
            params["document_store"] = Db2DocumentStore.from_dict(params["document_store"])
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := params.get("filter_policy"):
            params["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
