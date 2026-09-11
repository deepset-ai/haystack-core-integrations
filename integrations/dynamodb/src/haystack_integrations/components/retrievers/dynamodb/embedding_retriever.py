# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore
from haystack_integrations.document_stores.dynamodb.document_store import SEARCH_VECTORS_MAX_TOP_K


@component
class DynamoDBEmbeddingRetriever:
    """
    Retrieves documents from a `DynamoDBDocumentStore` using vector similarity on embeddings.

    Uses DynamoDB's native `SearchVectors` API (cosine similarity). DynamoDB returns at most 100
    candidates per search, so `top_k` cannot exceed 100. Metadata filters are applied client-side
    to those candidates, so a selective filter can return fewer than `top_k` documents even when
    more matching documents exist.

    Example usage:

    ```python
    from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore
    from haystack_integrations.components.retrievers.dynamodb import DynamoDBEmbeddingRetriever

    store = DynamoDBDocumentStore(table_name="docs", index_name="doc-index", embedding_dimension=768)
    retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5)
    result = retriever.run(query_embedding=[0.1, 0.2, ...])
    ```
    """

    def __init__(
        self,
        *,
        document_store: DynamoDBDocumentStore,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Creates a new DynamoDBEmbeddingRetriever.

        :param document_store: The `DynamoDBDocumentStore` to retrieve documents from.
        :param top_k: Maximum number of documents to return, between 1 and 100 (the DynamoDB
            `SearchVectors` limit).
        :param filters: Optional Haystack metadata filters applied at retrieval time. Applied
            client-side after the native vector search, since DynamoDB's `SearchVectors`
            filter expressions can only reference attributes declared in the index's
            `SearchSchema` at index-creation time.
        :param filter_policy: How run-time filters combine with `filters`: `REPLACE` (default)
            uses the run-time filters alone when they are given, `MERGE` combines both.
        :raises ValueError: If `document_store` is not a `DynamoDBDocumentStore` or `top_k` is
            outside the allowed range.
        """
        if not isinstance(document_store, DynamoDBDocumentStore):
            msg = f"document_store must be a DynamoDBDocumentStore, got {type(document_store)}"
            raise ValueError(msg)
        if not 1 <= top_k <= SEARCH_VECTORS_MAX_TOP_K:
            msg = f"top_k must be between 1 and {SEARCH_VECTORS_MAX_TOP_K} (DynamoDB SearchVectors limit), got {top_k}."
            raise ValueError(msg)
        self.document_store = document_store
        self.top_k = top_k
        self.filters = filters
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents most similar to `query_embedding`.

        :param query_embedding: The query vector.
        :param top_k: Overrides the instance-level `top_k` for this call; must stay between 1 and 100.
        :param filters: Run-time filters, combined with the instance-level `filters` according to
            `filter_policy`.
        :returns: A dictionary with `documents`, a list of `Document` objects sorted by score.
        """
        top_k = top_k if top_k is not None else self.top_k
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store._embedding_retrieval(query_embedding=query_embedding, top_k=top_k, filters=filters)
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieves documents most similar to `query_embedding`.

        :param query_embedding: The query vector.
        :param top_k: Overrides the instance-level `top_k` for this call; must stay between 1 and 100.
        :param filters: Run-time filters, combined with the instance-level `filters` according to
            `filter_policy`.
        :returns: A dictionary with `documents`, a list of `Document` objects sorted by score.
        """
        top_k = top_k if top_k is not None else self.top_k
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = await self.document_store._embedding_retrieval_async(
            query_embedding=query_embedding, top_k=top_k, filters=filters
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
            top_k=self.top_k,
            filters=self.filters,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamoDBEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_params = data["init_parameters"]
        init_params["document_store"] = DynamoDBDocumentStore.from_dict(init_params["document_store"])
        if filter_policy := init_params.get("filter_policy"):
            init_params["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
