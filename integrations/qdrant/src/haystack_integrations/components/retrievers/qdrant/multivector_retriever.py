# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from qdrant_client import models

from haystack_integrations.components.retrievers.qdrant.utils import _resolve_filters
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


@component
class QdrantMultivectorEmbeddingRetriever:
    """
    A component for retrieving documents from a QdrantDocumentStore using multivector embeddings.

    Documents are ranked by dot-product MaxSim. Documents without multivector embeddings are excluded.
    Direct retrieval scans the filtered documents and can be computationally expensive on large collections.
    To limit MaxSim scoring to a smaller set of candidates, use `QdrantEmbeddingRetriever(return_embedding=True)`
    followed by `MaxSimRanker`.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack_integrations.components.retrievers.qdrant import QdrantMultivectorEmbeddingRetriever

    store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=2)
    store.write_documents([
        Document(id="solar", content="Solar energy.", multivector_embedding=[[1.0, 0.0], [0.0, 1.0]]),
        Document(id="wind", content="Wind energy.", multivector_embedding=[[0.5, 0.0]]),
    ])
    retriever = QdrantMultivectorEmbeddingRetriever(document_store=store, top_k=1)
    result = retriever.run(query_multivector_embedding=[[1.0, 0.0], [0.0, 1.0]])
    print(result["documents"][0].id, result["documents"][0].score)
    # >> solar 2.0
    retriever.close()
    ```
    """

    def __init__(
        self,
        document_store: QdrantDocumentStore,
        *,
        filters: dict[str, Any] | models.Filter | None = None,
        top_k: int = 10,
        return_embedding: bool = False,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create a QdrantMultivectorEmbeddingRetriever component.

        :param document_store: An instance of QdrantDocumentStore with `multivector_embedding_dim` configured.
        :param filters: Haystack or native Qdrant filters to narrow down the search space.
        :param top_k: The maximum number of documents to retrieve. Must be greater than zero.
        :param return_embedding: Whether to return all stored embeddings of the retrieved Documents.
        :param filter_policy: Policy to determine how filters are applied. Native Qdrant filters only support REPLACE.

        :raises TypeError: `document_store` is not an instance of `QdrantDocumentStore`.
        :raises ValueError: Multivector support is disabled, `top_k` is not positive, or `filter_policy` is invalid.
        """
        if not isinstance(document_store, QdrantDocumentStore):
            msg = "document_store must be a QdrantDocumentStore."
            raise TypeError(msg)
        if document_store.multivector_embedding_dim is None:
            msg = "Set multivector_embedding_dim on the Document Store before creating this retriever."
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero."
            raise ValueError(msg)
        self._document_store = document_store
        self._filters = filters
        self._top_k = top_k
        self._return_embedding = return_embedding
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
            filters=self._filters.model_dump(mode="json")
            if isinstance(self._filters, models.Filter)
            else self._filters,
            top_k=self._top_k,
            return_embedding=self._return_embedding,
            filter_policy=self._filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QdrantMultivectorEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data = dict(data)
        data["init_parameters"] = dict(data["init_parameters"])
        data["init_parameters"]["document_store"] = QdrantDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        filters = data["init_parameters"].get("filters")
        if isinstance(filters, dict) and "operator" not in filters and "must" in filters:
            data["init_parameters"]["filters"] = models.Filter.model_validate(filters)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_multivector_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | models.Filter | None = None,
        top_k: int | None = None,
        return_embedding: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Run the Multivector Embedding Retriever on the given input data.

        :param query_multivector_embedding: Multivector embedding of the query. Must be non-empty and match
            the Document Store's per-vector dimension.
        :param filters: Haystack or native Qdrant filters, applied according to `filter_policy`.
        :param top_k: The maximum number of documents to return. Must be greater than zero.
            If None, the initialization value is used.
        :param return_embedding: Whether to return all stored embeddings of the retrieved Documents.
            If None, the initialization value is used.
        :returns:
            A dictionary with `documents`: a list of Documents ordered by descending dot-product MaxSim score.

        :raises ValueError: The query is empty or has an incompatible dimension, `top_k` is not positive,
            or native Qdrant filters are used with MERGE.
        :raises QdrantStoreError: The backend query fails or the collection is incompatible.
        """
        documents = self._document_store._query_by_multivector_embedding(
            query_multivector_embedding=query_multivector_embedding,
            filters=_resolve_filters(
                filter_policy=self._filter_policy, init_filters=self._filters, runtime_filters=filters
            ),
            top_k=self._top_k if top_k is None else top_k,
            return_embedding=self._return_embedding if return_embedding is None else return_embedding,
        )
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_multivector_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | models.Filter | None = None,
        top_k: int | None = None,
        return_embedding: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously run the Multivector Embedding Retriever on the given input data.

        :param query_multivector_embedding: Multivector embedding of the query. Must be non-empty and match
            the Document Store's per-vector dimension.
        :param filters: Haystack or native Qdrant filters, applied according to `filter_policy`.
        :param top_k: The maximum number of documents to return. Must be greater than zero.
            If None, the initialization value is used.
        :param return_embedding: Whether to return all stored embeddings of the retrieved Documents.
            If None, the initialization value is used.
        :returns:
            A dictionary with `documents`: a list of Documents ordered by descending dot-product MaxSim score.

        :raises ValueError: The query is empty or has an incompatible dimension, `top_k` is not positive,
            or native Qdrant filters are used with MERGE.
        :raises QdrantStoreError: The backend query fails or the collection is incompatible.
        """
        documents = await self._document_store._query_by_multivector_embedding_async(
            query_multivector_embedding=query_multivector_embedding,
            filters=_resolve_filters(
                filter_policy=self._filter_policy, init_filters=self._filters, runtime_filters=filters
            ),
            top_k=self._top_k if top_k is None else top_k,
            return_embedding=self._return_embedding if return_embedding is None else return_embedding,
        )
        return {"documents": documents}

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
