# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from qdrant_client.http import models as rest

from haystack_integrations.components.retrievers.qdrant import QdrantMultivectorEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class TestQdrantMultivectorEmbeddingRetriever:
    def test_init_raises_when_document_store_is_not_qdrant(self):
        with pytest.raises(TypeError, match="document_store must be a QdrantDocumentStore"):
            QdrantMultivectorEmbeddingRetriever(document_store="not a document store")

    def test_init_raises_when_multivectors_are_disabled(self):
        document_store = QdrantDocumentStore(location=":memory:")
        with pytest.raises(ValueError, match="Set multivector_embedding_dim"):
            QdrantMultivectorEmbeddingRetriever(document_store=document_store)

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_init_raises_when_top_k_is_not_positive(self, top_k):
        document_store = QdrantDocumentStore(multivector_embedding_dim=2)
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            QdrantMultivectorEmbeddingRetriever(document_store=document_store, top_k=top_k)

    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=2)
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=document_store)
        assert retriever._document_store is document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._return_embedding is False
        assert retriever._filter_policy == FilterPolicy.REPLACE

    def test_init_filter_policy(self):
        document_store = QdrantDocumentStore(multivector_embedding_dim=2)
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=document_store, filter_policy="merge")
        assert retriever._filter_policy == FilterPolicy.MERGE
        with pytest.raises(ValueError):
            QdrantMultivectorEmbeddingRetriever(document_store=document_store, filter_policy="invalid")

    def test_to_dict(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test", multivector_embedding_dim=2)
        filters = {"field": "meta.kind", "operator": "==", "value": "A"}
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=document_store,
            filters=filters,
            top_k=5,
            return_embedding=True,
            filter_policy=FilterPolicy.MERGE,
        )
        assert retriever.to_dict() == {
            "type": (
                "haystack_integrations.components.retrievers.qdrant.multivector_retriever."
                "QdrantMultivectorEmbeddingRetriever"
            ),
            "init_parameters": {
                "document_store": document_store.to_dict(),
                "filters": filters,
                "top_k": 5,
                "return_embedding": True,
                "filter_policy": "merge",
            },
        }

    def test_from_dict(self):
        data = {
            "type": (
                "haystack_integrations.components.retrievers.qdrant.multivector_retriever."
                "QdrantMultivectorEmbeddingRetriever"
            ),
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                    "init_parameters": {"location": ":memory:", "index": "test", "multivector_embedding_dim": 2},
                },
                "filters": {"field": "meta.kind", "operator": "==", "value": "A"},
                "top_k": 5,
                "return_embedding": True,
                "filter_policy": "merge",
            },
        }
        retriever = QdrantMultivectorEmbeddingRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._document_store.multivector_embedding_dim == 2
        assert retriever._filters == {"field": "meta.kind", "operator": "==", "value": "A"}
        assert retriever._top_k == 5
        assert retriever._return_embedding is True
        assert retriever._filter_policy == FilterPolicy.MERGE

    def test_serialization_with_native_filters(self):
        document_store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=2)
        filters = rest.Filter(must=[rest.FieldCondition(key="meta.kind", match=rest.MatchValue(value="A"))])
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=document_store, filters=filters)
        data = retriever.to_dict()
        assert data["init_parameters"]["filters"] == filters.model_dump(mode="json")
        restored = QdrantMultivectorEmbeddingRetriever.from_dict(data)
        assert restored._filters == filters

    def test_close(self):
        mock_store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=mock_store)
        retriever.close()
        mock_store.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_close_async(self):
        mock_store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=mock_store)
        await retriever.close_async()
        mock_store.close_async.assert_awaited_once_with()

    def test_run(self):
        mock_store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        documents = [Document(id="a", score=2.0)]
        mock_store._query_by_multivector_embedding.return_value = documents
        filters = {"field": "meta.kind", "operator": "==", "value": "A"}
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=mock_store,
            filters=filters,
            top_k=7,
            return_embedding=True,
        )
        assert retriever.run(query_multivector_embedding=[[1.0, 0.0]]) == {"documents": documents}
        mock_store._query_by_multivector_embedding.assert_called_once_with(
            query_multivector_embedding=[[1.0, 0.0]],
            filters=filters,
            top_k=7,
            return_embedding=True,
        )

    @pytest.mark.asyncio
    async def test_run_async(self):
        mock_store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        documents = [Document(id="a", score=2.0)]
        mock_store._query_by_multivector_embedding_async = AsyncMock(return_value=documents)
        filters = {"field": "meta.kind", "operator": "==", "value": "A"}
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=mock_store,
            filters=filters,
            top_k=7,
            return_embedding=True,
        )
        assert await retriever.run_async(query_multivector_embedding=[[1.0, 0.0]]) == {"documents": documents}
        mock_store._query_by_multivector_embedding_async.assert_awaited_once_with(
            query_multivector_embedding=[[1.0, 0.0]],
            filters=filters,
            top_k=7,
            return_embedding=True,
        )

    def test_run_runtime_values_override_init(self):
        mock_store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        mock_store._query_by_multivector_embedding.return_value = []
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=mock_store,
            top_k=7,
            return_embedding=True,
            filters={"field": "meta.kind", "operator": "==", "value": "A"},
        )
        filters = rest.Filter(must=[])
        retriever.run(query_multivector_embedding=[[1.0, 0.0]], top_k=3, return_embedding=False, filters=filters)
        mock_store._query_by_multivector_embedding.assert_called_once_with(
            query_multivector_embedding=[[1.0, 0.0]],
            filters=filters,
            top_k=3,
            return_embedding=False,
        )

    @pytest.mark.asyncio
    async def test_run_async_runtime_values_override_init(self):
        mock_store = Mock(spec=QdrantDocumentStore, multivector_embedding_dim=2)
        mock_store._query_by_multivector_embedding_async = AsyncMock(return_value=[])
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=mock_store,
            top_k=7,
            return_embedding=True,
            filters={"field": "meta.kind", "operator": "==", "value": "A"},
        )
        filters = rest.Filter(must=[])
        await retriever.run_async(
            query_multivector_embedding=[[1.0, 0.0]], top_k=3, return_embedding=False, filters=filters
        )
        mock_store._query_by_multivector_embedding_async.assert_awaited_once_with(
            query_multivector_embedding=[[1.0, 0.0]],
            filters=filters,
            top_k=3,
            return_embedding=False,
        )


@pytest.mark.integration
class TestQdrantMultivectorEmbeddingRetrieverIntegration:
    @pytest.fixture
    def documents(self):
        return [
            Document(
                id="a",
                embedding=[1.0, 0.0],
                sparse_embedding=SparseEmbedding(indices=[1], values=[0.5]),
                multivector_embedding=[[2.0, 0.0], [0.0, 3.0]],
                meta={"kind": "A", "page": 1},
            ),
            Document(id="b", multivector_embedding=[[1.0, 1.0]], meta={"kind": "B", "page": 2}),
            Document(id="c", multivector_embedding=[[0.5, 0.0]], meta={"kind": "A", "page": 2}),
            Document(id="no-multivector", embedding=[1.0, 0.0]),
        ]

    def test_run(self, documents):
        document_store = QdrantDocumentStore(
            location=":memory:",
            embedding_dim=2,
            multivector_embedding_dim=2,
            use_sparse_embeddings=True,
        )
        document_store.write_documents(documents)
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=document_store)
        result = retriever.run(query_multivector_embedding=[[1.0, 0.0], [0.0, 1.0]])["documents"]
        assert [doc.id for doc in result] == ["a", "b", "c"]
        assert [doc.score for doc in result] == pytest.approx([5.0, 2.0, 0.5])
        assert all(doc.multivector_embedding is None for doc in result)

        result = retriever.run(query_multivector_embedding=[[1.0, 0.0]], top_k=1, return_embedding=True)["documents"]
        assert [doc.id for doc in result] == ["a"]
        assert result[0].multivector_embedding == documents[0].multivector_embedding
        assert result[0].embedding == documents[0].embedding
        assert result[0].sparse_embedding == documents[0].sparse_embedding
        retriever.close()

    @pytest.mark.asyncio
    async def test_run_async(self, documents):
        document_store = QdrantDocumentStore(
            location=":memory:",
            embedding_dim=2,
            multivector_embedding_dim=2,
            use_sparse_embeddings=True,
        )
        await document_store.write_documents_async(documents)
        retriever = QdrantMultivectorEmbeddingRetriever(document_store=document_store)
        result = (await retriever.run_async(query_multivector_embedding=[[1.0, 0.0], [0.0, 1.0]]))["documents"]
        assert [doc.id for doc in result] == ["a", "b", "c"]
        assert [doc.score for doc in result] == pytest.approx([5.0, 2.0, 0.5])
        assert all(doc.multivector_embedding is None for doc in result)

        result = (await retriever.run_async(query_multivector_embedding=[[1.0, 0.0]], top_k=1, return_embedding=True))[
            "documents"
        ]
        assert [doc.id for doc in result] == ["a"]
        assert result[0].multivector_embedding == documents[0].multivector_embedding
        assert result[0].embedding == documents[0].embedding
        assert result[0].sparse_embedding == documents[0].sparse_embedding
        await retriever.close_async()

    def test_run_filters(self, documents):
        document_store = QdrantDocumentStore(
            location=":memory:", embedding_dim=2, multivector_embedding_dim=2, use_sparse_embeddings=True
        )
        document_store.write_documents(documents)
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.kind", "operator": "==", "value": "A"},
            filter_policy=FilterPolicy.MERGE,
        )
        result = retriever.run(query_multivector_embedding=[[1.0, 0.0]])["documents"]
        assert [doc.id for doc in result] == ["a", "c"]
        result = retriever.run(
            query_multivector_embedding=[[1.0, 0.0]], filters={"field": "meta.page", "operator": "==", "value": 2}
        )["documents"]
        assert [doc.id for doc in result] == ["c"]
        retriever.close()

    @pytest.mark.asyncio
    async def test_run_filters_async(self, documents):
        document_store = QdrantDocumentStore(
            location=":memory:", embedding_dim=2, multivector_embedding_dim=2, use_sparse_embeddings=True
        )
        await document_store.write_documents_async(documents)
        retriever = QdrantMultivectorEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.kind", "operator": "==", "value": "A"},
            filter_policy=FilterPolicy.MERGE,
        )
        result = (await retriever.run_async(query_multivector_embedding=[[1.0, 0.0]]))["documents"]
        assert [doc.id for doc in result] == ["a", "c"]
        result = (
            await retriever.run_async(
                query_multivector_embedding=[[1.0, 0.0]], filters={"field": "meta.page", "operator": "==", "value": 2}
            )
        )["documents"]
        assert [doc.id for doc in result] == ["c"]
        await retriever.close_async()
