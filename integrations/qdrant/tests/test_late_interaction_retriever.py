# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from haystack.utils import Secret
from qdrant_client.http import models

from haystack_integrations.components.retrievers.qdrant import QdrantLateInteractionRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.qdrant.converters import (
    DENSE_VECTORS_NAME,
    MULTIVECTOR_NAME,
    SPARSE_VECTORS_NAME,
)
from haystack_integrations.document_stores.qdrant.filters import convert_filters_to_qdrant


class TestQdrantLateInteractionRetriever:
    def test_init_defaults(self):
        store = QdrantDocumentStore(embedding_dim=2, multivector_embedding_dim=2)
        retriever = QdrantLateInteractionRetriever(store)
        assert retriever._document_store is store
        assert retriever._top_k == 10
        assert retriever._prefetch_top_k == 100
        assert retriever._fusion_top_k is None
        assert retriever._filters is None
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert store._client is None
        assert store._async_client is None

    @pytest.mark.parametrize(
        "store, error, match",
        [
            ("not a store", ValueError, "document_store must be"),
            (QdrantDocumentStore(), ValueError, "Set multivector_embedding_dim"),
        ],
    )
    def test_invalid_store(self, store, error, match):
        with pytest.raises(error, match=match):
            QdrantLateInteractionRetriever(store)

    @pytest.mark.parametrize(
        "filters, policy",
        [
            ({"field": "meta.kind", "operator": "==", "value": "A"}, "merge"),
            (
                models.Filter(must=[models.FieldCondition(key="meta.kind", match=models.MatchValue(value="A"))]),
                "replace",
            ),
        ],
    )
    def test_serialization(self, monkeypatch, filters, policy):
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        store = QdrantDocumentStore(
            location=":memory:",
            index="late-interaction",
            embedding_dim=2,
            multivector_embedding_dim=2,
            use_sparse_embeddings=True,
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
        )
        retriever = QdrantLateInteractionRetriever(
            store, filters=filters, top_k=3, prefetch_top_k=50, fusion_top_k=20, filter_policy=policy
        )
        data = retriever.to_dict()
        original = deepcopy(data)
        restored = QdrantLateInteractionRetriever.from_dict(data)
        assert data == original
        assert restored.to_dict() == data
        assert restored._filters == filters
        assert restored._document_store.api_key.resolve_value() == "test-key"
        assert data["init_parameters"] == {
            "document_store": store.to_dict(),
            "filters": filters.model_dump(mode="json") if isinstance(filters, models.Filter) else filters,
            "top_k": 3,
            "prefetch_top_k": 50,
            "fusion_top_k": 20,
            "filter_policy": policy,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize(
        "use_sparse, fusion_top_k, override",
        [(False, None, False), (False, 25, True), (True, None, True), (True, 25, False)],
        ids=["dense-union", "dense-fusion-override", "hybrid-union-override", "hybrid-fusion"],
    )
    async def test_query_request(self, use_async, use_sparse, fusion_top_k, override):
        store = QdrantDocumentStore(embedding_dim=2, multivector_embedding_dim=2, use_sparse_embeddings=use_sparse)
        client = AsyncMock() if use_async else Mock()
        client.query_points.return_value = models.QueryResponse(
            points=[models.ScoredPoint(id=1, version=0, score=2.5, payload={"id": "a", "content": "A"})]
        )
        if use_async:
            store._async_client = client
        else:
            store._client = client
        init_filters = {"field": "meta.kind", "operator": "==", "value": "A"}
        runtime_filters = models.Filter(must=[])
        retriever = QdrantLateInteractionRetriever(
            store, filters=init_filters, top_k=7, prefetch_top_k=50, fusion_top_k=fusion_top_k
        )
        kwargs = {"query_embedding": [1.0, 0.0], "query_multivector_embedding": [[1.0, 0.0]]}
        if use_sparse:
            kwargs["query_sparse_embedding"] = SparseEmbedding(indices=[1], values=[0.8])
        if override:
            kwargs.update(top_k=3, prefetch_top_k=40, filters=runtime_filters)
        result = await retriever.run_async(**kwargs) if use_async else retriever.run(**kwargs)
        assert result == {"documents": [Document(id="a", content="A", score=2.5)]}
        query_filter = runtime_filters if override else convert_filters_to_qdrant(init_filters)
        prefetch = [
            models.Prefetch(
                query=[1.0, 0.0], using=DENSE_VECTORS_NAME, limit=40 if override else 50, filter=query_filter
            )
        ]
        if use_sparse:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=[1], values=[0.8]),
                    using=SPARSE_VECTORS_NAME,
                    limit=40 if override else 50,
                    filter=query_filter,
                )
            )
        if fusion_top_k is not None:
            prefetch = [
                models.Prefetch(
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.DBSF),
                    limit=fusion_top_k,
                    filter=query_filter,
                )
            ]
        expected = {
            "collection_name": store.index,
            "query": [[1.0, 0.0]],
            "using": MULTIVECTOR_NAME,
            "prefetch": prefetch,
            "query_filter": query_filter,
            "limit": 3 if override else 7,
            "with_payload": True,
            "with_vectors": False,
        }
        client.query_points.assert_called_once_with(**expected)
        if use_async:
            client.query_points.assert_awaited_once_with(**expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            ({"query_embedding": []}, ValueError, "embedding_dim"),
            ({"query_embedding": [1.0]}, ValueError, "embedding_dim"),
            (
                {"query_sparse_embedding": SparseEmbedding(indices=[1], values=[1.0])},
                ValueError,
                "use_sparse_embeddings",
            ),
        ],
    )
    async def test_invalid_run_inputs(self, use_async, kwargs, error, match):
        store = QdrantDocumentStore(embedding_dim=2, multivector_embedding_dim=2)
        retriever = QdrantLateInteractionRetriever(store)
        inputs = {"query_embedding": [1.0, 0.0], "query_multivector_embedding": [[1.0, 0.0]], **kwargs}
        with pytest.raises(error, match=match):
            if use_async:
                await retriever.run_async(**inputs)
            else:
                retriever.run(**inputs)
        assert store._client is None
        assert store._async_client is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_async", [False, True])
    async def test_close(self, use_async):
        store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        retriever = QdrantLateInteractionRetriever(store)
        if use_async:
            await retriever.close_async()
            store.close_async.assert_awaited_once_with()
        else:
            retriever.close()
            store.close.assert_called_once_with()


@pytest.mark.integration
@pytest.mark.asyncio
class TestQdrantLateInteractionRetrieverIntegration:
    @pytest.fixture
    def documents(self):
        return [
            Document(
                id="dense",
                content="Dense match",
                embedding=[1.0, 0.0],
                sparse_embedding=SparseEmbedding(indices=[2], values=[1.0]),
                multivector_embedding=[[1.0, 0.0]],
                meta={"kind": "A", "page": 1},
            ),
            Document(
                id="sparse",
                content="Sparse match",
                embedding=[0.0, 1.0],
                sparse_embedding=SparseEmbedding(indices=[1], values=[3.0]),
                multivector_embedding=[[3.0, 0.0]],
                meta={"kind": "B", "page": 2},
            ),
            Document(
                id="both",
                content="Both match",
                embedding=[0.8, 0.2],
                sparse_embedding=SparseEmbedding(indices=[1], values=[2.0]),
                multivector_embedding=[[2.0, 0.0]],
                meta={"kind": "A", "page": 2},
            ),
            Document(
                id="outside",
                content="Outside shortlist",
                embedding=[-1.0, 0.0],
                sparse_embedding=SparseEmbedding(indices=[2], values=[1.0]),
                multivector_embedding=[[10.0, 0.0]],
                meta={"kind": "C", "page": 2},
            ),
        ]

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize(
        "sparse_enabled, use_sparse, fusion_top_k",
        [(False, False, None), (False, False, 1), (True, False, None), (True, True, None), (True, True, 1)],
        ids=["dense-union", "dense-fusion", "sparse-enabled-dense-only", "hybrid-union", "hybrid-fusion"],
    )
    async def test_reranking(self, documents, use_async, sparse_enabled, use_sparse, fusion_top_k):
        store = QdrantDocumentStore(
            location=":memory:", embedding_dim=2, multivector_embedding_dim=2, use_sparse_embeddings=sparse_enabled
        )
        if not sparse_enabled:
            documents = [replace(doc, sparse_embedding=None) for doc in documents]
        retriever = QdrantLateInteractionRetriever(store, prefetch_top_k=2, fusion_top_k=fusion_top_k)
        kwargs = {"query_embedding": [1.0, 0.0], "query_multivector_embedding": [[1.0, 0.0]]}
        if use_sparse:
            kwargs["query_sparse_embedding"] = SparseEmbedding(indices=[1], values=[1.0])
        if use_async:
            await store.write_documents_async(documents)
            result = (await retriever.run_async(**kwargs))["documents"]
        else:
            store.write_documents(documents)
            result = retriever.run(**kwargs)["documents"]
        if fusion_top_k is not None:
            expected_ids, expected_scores = (["both"], [2.0]) if use_sparse else (["dense"], [1.0])
        elif use_sparse:
            expected_ids, expected_scores = ["sparse", "both", "dense"], [3.0, 2.0, 1.0]
        else:
            expected_ids, expected_scores = ["both", "dense"], [2.0, 1.0]
        assert [doc.id for doc in result] == expected_ids
        assert [doc.score for doc in result] == pytest.approx(expected_scores)
        assert all(doc.content for doc in result)
        assert all(
            doc.embedding is None and doc.sparse_embedding is None and doc.multivector_embedding is None
            for doc in result
        )
        if use_async:
            await retriever.close_async()
        else:
            retriever.close()

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize("fusion_top_k", [None, 1])
    async def test_filters_before_candidate_selection(self, documents, use_async, fusion_top_k):
        documents = [
            *documents,
            Document(
                id="excluded",
                embedding=[1.0, 0.0],
                sparse_embedding=SparseEmbedding(indices=[1], values=[20.0]),
                multivector_embedding=[[20.0, 0.0]],
                meta={"kind": "C", "page": 1},
            ),
        ]
        store = QdrantDocumentStore(
            location=":memory:", embedding_dim=2, multivector_embedding_dim=2, use_sparse_embeddings=True
        )
        retriever = QdrantLateInteractionRetriever(
            store,
            prefetch_top_k=1,
            fusion_top_k=fusion_top_k,
            filters={"field": "meta.page", "operator": "==", "value": 2},
            filter_policy="merge",
        )
        kwargs = {
            "query_embedding": [1.0, 0.0],
            "query_multivector_embedding": [[1.0, 0.0]],
            "query_sparse_embedding": SparseEmbedding(indices=[1], values=[1.0]),
            "filters": {"field": "meta.kind", "operator": "==", "value": "C"},
        }
        if use_async:
            await store.write_documents_async(documents)
            result = (await retriever.run_async(**kwargs))["documents"]
            empty = (
                await retriever.run_async(**{**kwargs, "filters": {"field": "meta.page", "operator": "==", "value": 3}})
            )["documents"]
            await retriever.close_async()
        else:
            store.write_documents(documents)
            result = retriever.run(**kwargs)["documents"]
            empty = retriever.run(**{**kwargs, "filters": {"field": "meta.page", "operator": "==", "value": 3}})[
                "documents"
            ]
            retriever.close()
        assert [doc.id for doc in result] == ["outside"]
        assert result[0].score == pytest.approx(10.0)
        assert result[0].meta == {"kind": "C", "page": 2}
        assert empty == []
