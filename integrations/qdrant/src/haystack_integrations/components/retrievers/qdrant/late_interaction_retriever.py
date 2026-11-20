# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from qdrant_client import models

from haystack_integrations.components.retrievers.qdrant.utils import (
    _deserialize_filters,
    _resolve_filters,
    _serialize_filters,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.qdrant.converters import DENSE_VECTORS_NAME, SPARSE_VECTORS_NAME
from haystack_integrations.document_stores.qdrant.filters import convert_filters_to_qdrant


@component
class QdrantLateInteractionRetriever:
    """
    Retrieve dense and optional sparse candidates, then rerank with MaxSim inside Qdrant.

    A single query request retrieves up to `prefetch_top_k` candidates per branch and scores their union
    against the query's token vectors. Set `fusion_top_k` to first fuse the candidates with DBSF and cap
    the shortlist. Final scores are dot-product MaxSim scores, not fusion scores. Documents without
    multivector embeddings are excluded. Only final documents and scores are returned, never vectors.

    Sparse embeddings are optional, including for stores with sparse support enabled. For BM25 retrieval,
    supply compatible BM25 document/query embeddings and configure the store's `sparse_idf` as required
    by your embedding model; this component does not generate embeddings. Larger candidate sets improve
    recall but increase MaxSim computation. See [Qdrant multi-stage queries](https://qdrant.tech/documentation/hybrid-queries/).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack_integrations.components.retrievers.qdrant import QdrantLateInteractionRetriever

    store = QdrantDocumentStore(location=":memory:", embedding_dim=2, multivector_embedding_dim=2)
    store.write_documents([
        Document(id="solar", content="Solar energy.", embedding=[1.0, 0.0],
                 multivector_embedding=[[1.0, 0.0], [0.0, 1.0]]),
        Document(id="wind", content="Wind energy.", embedding=[0.8, 0.2],
                 multivector_embedding=[[0.5, 0.0]]),
    ])
    retriever = QdrantLateInteractionRetriever(document_store=store, prefetch_top_k=100, top_k=1)
    result = retriever.run(query_embedding=[1.0, 0.0],
                           query_multivector_embedding=[[1.0, 0.0], [0.0, 1.0]])
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
        prefetch_top_k: int = 100,
        fusion_top_k: int | None = None,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create a QdrantLateInteractionRetriever.

        :param document_store: Qdrant store with `multivector_embedding_dim` configured. Sparse support is
            required only when supplying a sparse query embedding.
        :param filters: Haystack or native Qdrant filters applied before candidate selection and reranking.
        :param top_k: Maximum number of final documents. Must be a positive integer.
        :param prefetch_top_k: Maximum candidates per dense/sparse branch. Must be a positive integer.
        :param fusion_top_k: Positive DBSF shortlist limit, or None to rerank the full candidate union.
            With a dense-only query, this optionally caps the dense shortlist before MaxSim.
        :param filter_policy: How runtime filters are applied. Native Qdrant filters support only REPLACE.
        :raises ValueError: If `document_store` is not an instance of `QdrantDocumentStore` or
            Multivector support is disabled or the filter policy is invalid.
        """
        if not isinstance(document_store, QdrantDocumentStore):
            msg = "document_store must be an instance of QdrantDocumentStore"
            raise ValueError(msg)
        if document_store.multivector_embedding_dim is None:
            msg = "Set multivector_embedding_dim on the Document Store before creating this retriever."
            raise ValueError(msg)
        self._document_store = document_store
        self._filters = filters
        self._top_k = top_k
        self._prefetch_top_k = prefetch_top_k
        self._fusion_top_k = fusion_top_k
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary containing the component type and initialization parameters.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=_serialize_filters(self._filters),
            top_k=self._top_k,
            prefetch_top_k=self._prefetch_top_k,
            fusion_top_k=self._fusion_top_k,
            filter_policy=self._filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QdrantLateInteractionRetriever":
        """
        Deserialize the component from a dictionary.

        :param data: Serialized component data.
        :returns: Deserialized component.
        """
        data = dict(data)
        data["init_parameters"] = dict(data["init_parameters"])
        data["init_parameters"]["document_store"] = QdrantDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        data["init_parameters"]["filters"] = _deserialize_filters(data["init_parameters"].get("filters"))
        return default_from_dict(cls, data)

    def _prepare_prefetch(
        self,
        query_embedding: list[float],
        query_sparse_embedding: SparseEmbedding | None,
        filters: dict[str, Any] | models.Filter | None,
        top_k: int | None,
        prefetch_top_k: int | None,
    ) -> tuple[list[models.Prefetch], models.Filter | None, int]:
        top_k = self._top_k if top_k is None else top_k
        prefetch_top_k = self._prefetch_top_k if prefetch_top_k is None else prefetch_top_k
        if not query_embedding or len(query_embedding) != self._document_store.embedding_dim:
            msg = "query_embedding must be non-empty and match the Document Store's embedding_dim."
            raise ValueError(msg)
        if query_sparse_embedding is not None and not self._document_store.use_sparse_embeddings:
            msg = "Set use_sparse_embeddings=True on the Document Store before supplying a sparse query."
            raise ValueError(msg)
        filters = _resolve_filters(
            filter_policy=self._filter_policy, init_filters=self._filters, runtime_filters=filters
        )
        qdrant_filters = convert_filters_to_qdrant(filters)
        prefetch = [
            models.Prefetch(
                query=query_embedding, using=DENSE_VECTORS_NAME, limit=prefetch_top_k, filter=qdrant_filters
            )
        ]
        if query_sparse_embedding is not None:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse_embedding.indices, values=query_sparse_embedding.values
                    ),
                    using=SPARSE_VECTORS_NAME,
                    limit=prefetch_top_k,
                    filter=qdrant_filters,
                )
            )
        if self._fusion_top_k is not None:
            prefetch = [
                models.Prefetch(
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.DBSF),
                    limit=self._fusion_top_k,
                    filter=qdrant_filters,
                )
            ]
        return prefetch, qdrant_filters, top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        query_multivector_embedding: list[list[float]],
        *,
        query_sparse_embedding: SparseEmbedding | None = None,
        filters: dict[str, Any] | models.Filter | None = None,
        top_k: int | None = None,
        prefetch_top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve candidates and rerank them in a single Qdrant query request.

        :param query_embedding: Dense query vector matching the Document Store's `embedding_dim`.
        :param query_multivector_embedding: Non-empty token matrix matching the store's per-token dimension.
        :param query_sparse_embedding: Optional sparse query vector, such as BM25. None uses dense candidates only.
        :param filters: Filters applied at every query stage according to `filter_policy`.
        :param top_k: Positive final result limit, or None to use the initialization value.
        :param prefetch_top_k: Positive per-branch candidate limit, or None to use the initialization value.
        :returns: A dictionary with `documents`: a list of Documents ordered by descending MaxSim, without embeddings.
        :raises TypeError: A dense/sparse query or limit has the wrong type.
        :raises ValueError: A query dimension or limit is invalid, sparse support is disabled for a sparse query,
            or native filters are used with MERGE.
        :raises QdrantStoreError: The backend query fails or the collection is incompatible.
        """
        prefetch, qdrant_filters, top_k = self._prepare_prefetch(
            query_embedding, query_sparse_embedding, filters, top_k, prefetch_top_k
        )
        documents = self._document_store._query_by_multivector_embedding(
            query_multivector_embedding=query_multivector_embedding,
            filters=qdrant_filters,
            top_k=top_k,
            return_embedding=False,
            prefetch=prefetch,
        )
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        query_multivector_embedding: list[list[float]],
        *,
        query_sparse_embedding: SparseEmbedding | None = None,
        filters: dict[str, Any] | models.Filter | None = None,
        top_k: int | None = None,
        prefetch_top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve candidates and rerank them in a single Qdrant query request.

        :param query_embedding: Dense query vector matching the Document Store's `embedding_dim`.
        :param query_multivector_embedding: Non-empty token matrix matching the store's per-token dimension.
        :param query_sparse_embedding: Optional sparse query vector, such as BM25. None uses dense candidates only.
        :param filters: Filters applied at every query stage according to `filter_policy`.
        :param top_k: Positive final result limit, or None to use the initialization value.
        :param prefetch_top_k: Positive per-branch candidate limit, or None to use the initialization value.
        :returns: A dictionary with `documents`: a list of Documents ordered by descending MaxSim, without embeddings.
        :raises TypeError: A dense/sparse query or limit has the wrong type.
        :raises ValueError: A query dimension or limit is invalid, sparse support is disabled for a sparse query,
            or native filters are used with MERGE.
        :raises QdrantStoreError: The backend query fails or the collection is incompatible.
        """
        prefetch, qdrant_filters, top_k = self._prepare_prefetch(
            query_embedding, query_sparse_embedding, filters, top_k, prefetch_top_k
        )
        documents = await self._document_store._query_by_multivector_embedding_async(
            query_multivector_embedding=query_multivector_embedding,
            filters=qdrant_filters,
            top_k=top_k,
            return_embedding=False,
            prefetch=prefetch,
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
