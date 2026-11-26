# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

from .utils import _resolve_document_store

logger = logging.getLogger(__name__)


@component
class OpenSearchMultivectorEmbeddingRetriever:
    """
    Retrieves documents from the OpenSearchDocumentStore using multivector embeddings.

    Documents are ranked using OpenSearch's `lateInteractionScore` with `innerproduct` similarity.
    Documents without multivector embeddings are excluded. Requires OpenSearch 3.3 or later.
    Direct retrieval scans the filtered documents and can be computationally expensive on large collections.
    To limit MaxSim scoring to a smaller set of candidates, use `OpenSearchEmbeddingRetriever` with
    `return_embedding=True` on the Document Store, followed by `MaxSimRanker`.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    from haystack_integrations.components.retrievers.opensearch import OpenSearchMultivectorEmbeddingRetriever

    store = OpenSearchDocumentStore(hosts="http://localhost:9200", multivector_embedding_dim=2)
    store.write_documents([
        Document(id="solar", content="Solar energy.", multivector_embedding=[[1.0, 0.0], [0.0, 1.0]]),
        Document(id="wind", content="Wind energy.", multivector_embedding=[[0.5, 0.0]]),
    ])
    retriever = OpenSearchMultivectorEmbeddingRetriever(document_store=store, top_k=1)
    result = retriever.run(query_multivector_embedding=[[1.0, 0.0], [0.0, 1.0]])
    print(result["documents"][0].id, result["documents"][0].score)
    # >> solar 4.0
    ```
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Create the OpenSearchMultivectorEmbeddingRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore with `multivector_embedding_dim` configured.
            Its `return_embedding` setting controls whether embeddings are returned.
        :param filters: Filters applied when fetching documents from the Document Store.
            Filters restrict the documents scored by OpenSearch.
        :param top_k: Maximum number of documents to return. Must be greater than zero.
        :param filter_policy: Policy to determine how filters are applied. Possible options:
        - `merge`: Runtime filters are merged with initialization filters.
        - `replace`: Runtime filters replace initialization filters. Use this policy to change the filtering scope.
        :param raise_on_failure: Whether to raise retrieval errors. If False, logs a warning and returns an empty list.

        :raises TypeError: `document_store` is not an instance of `OpenSearchDocumentStore`.
        :raises ValueError: Multivector support is disabled, `top_k` is not positive, or the filter policy is invalid.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise TypeError(msg)
        if document_store._multivector_embedding_dim is None:
            msg = "Set multivector_embedding_dim on the document store before creating this retriever."
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero."
            raise ValueError(msg)
        self._document_store = document_store
        self._filters = filters or {}
        self._raise_on_failure = raise_on_failure
        self._top_k = top_k
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
            raise_on_failure=self._raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenSearchMultivectorEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = OpenSearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    def close(self) -> None:
        """
        Release the synchronous resources of the underlying Document Store.
        """
        self._document_store.close()

    async def close_async(self) -> None:
        """
        Release the asynchronous resources of the underlying Document Store.
        """
        await self._document_store.close_async()

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_multivector_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        document_store: OpenSearchDocumentStore | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents using multivector embeddings.

        :param query_multivector_embedding: Multivector embedding of the query. Must be non-empty and match
            the Document Store's per-vector dimension.
        :param filters: Filters applied when fetching documents from the Document Store.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return. Must be greater than zero.
            If None, the initialization value is used.
        :param document_store: Optional instance of OpenSearchDocumentStore to use with the Retriever.
            If None, the Document Store provided at initialization is used.
        :returns:
            Dictionary with key "documents" containing the retrieved Documents.
            - documents: List of Documents sorted by descending native OpenSearch score.

        :raises ValueError: The runtime document store is invalid, or `raise_on_failure` is True and the embedding,
            multivector configuration, or `top_k` is invalid.
        """
        store = _resolve_document_store(document_store, self._document_store)
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        try:
            documents = store._multivector_retrieval(
                query_multivector_embedding=query_multivector_embedding,
                filters=filters,
                top_k=self._top_k if top_k is None else top_k,
            )
        except Exception as error:
            if self._raise_on_failure:
                raise
            logger.warning(
                "An error during multivector retrieval occurred and will be ignored by returning empty results: "
                "{error}",
                error=str(error),
                exc_info=True,
            )
            documents = []
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_multivector_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        document_store: OpenSearchDocumentStore | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents using multivector embeddings.

        :param query_multivector_embedding: Multivector embedding of the query. Must be non-empty and match
            the Document Store's per-vector dimension.
        :param filters: Filters applied when fetching documents from the Document Store.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return. Must be greater than zero.
            If None, the initialization value is used.
        :param document_store: Optional instance of OpenSearchDocumentStore to use with the Retriever.
            If None, the Document Store provided at initialization is used.
        :returns:
            Dictionary with key "documents" containing the retrieved Documents.
            - documents: List of Documents sorted by descending native OpenSearch score.

        :raises ValueError: The runtime document store is invalid, or `raise_on_failure` is True and the embedding,
            multivector configuration, or `top_k` is invalid.
        """
        store = _resolve_document_store(document_store, self._document_store)
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        try:
            documents = await store._multivector_retrieval_async(
                query_multivector_embedding=query_multivector_embedding,
                filters=filters,
                top_k=self._top_k if top_k is None else top_k,
            )
        except Exception as error:
            if self._raise_on_failure:
                raise
            logger.warning(
                "An error during multivector retrieval occurred and will be ignored by returning empty results: "
                "{error}",
                error=str(error),
                exc_info=True,
            )
            documents = []
        return {"documents": documents}
