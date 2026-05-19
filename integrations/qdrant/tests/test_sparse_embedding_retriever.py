from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from haystack.testing.document_store import (
    FilterableDocsFixtureMixin,
)
from qdrant_client.http import models as rest

from haystack_integrations.components.retrievers.qdrant import (
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class TestQdrantSparseEmbeddingRetriever:
    def test_init_raises_when_document_store_is_not_qdrant(self):
        with pytest.raises(ValueError, match="must be an instance of QdrantDocumentStore"):
            QdrantSparseEmbeddingRetriever(document_store="not a document store")

    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        assert retriever._document_store == document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._return_embedding is False
        assert retriever._score_threshold is None
        assert retriever._group_by is None
        assert retriever._group_size is None

        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store, filter_policy="replace")
        assert retriever._filter_policy == FilterPolicy.REPLACE

        with pytest.raises(ValueError):
            QdrantSparseEmbeddingRetriever(document_store=document_store, filter_policy="invalid")

    def test_to_dict(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        res = retriever.to_dict()
        assert res == {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                    "init_parameters": {
                        "location": ":memory:",
                        "url": None,
                        "port": 6333,
                        "grpc_port": 6334,
                        "prefer_grpc": False,
                        "https": None,
                        "api_key": None,
                        "prefix": None,
                        "timeout": None,
                        "host": None,
                        "path": None,
                        "index": "test",
                        "embedding_dim": 768,
                        "on_disk": False,
                        "force_disable_check_same_thread": False,
                        "use_sparse_embeddings": False,
                        "sparse_idf": False,
                        "similarity": "cosine",
                        "return_embedding": False,
                        "progress_bar": True,
                        "recreate_index": False,
                        "shard_number": None,
                        "replication_factor": None,
                        "write_consistency_factor": None,
                        "on_disk_payload": None,
                        "hnsw_config": None,
                        "optimizers_config": None,
                        "wal_config": None,
                        "quantization_config": None,
                        "wait_result_from_api": True,
                        "metadata": {},
                        "write_batch_size": 100,
                        "scroll_size": 10000,
                        "payload_fields_to_index": None,
                    },
                },
                "filters": None,
                "top_k": 10,
                "scale_score": False,
                "return_embedding": False,
                "filter_policy": "replace",
                "score_threshold": None,
                "group_by": None,
                "group_size": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "init_parameters": {"location": ":memory:", "index": "test"},
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                },
                "filters": None,
                "top_k": 5,
                "scale_score": False,
                "return_embedding": True,
                "filter_policy": "replace",
                "score_threshold": None,
                "group_by": None,
                "group_size": None,
            },
        }
        retriever = QdrantSparseEmbeddingRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._scale_score is False
        assert retriever._return_embedding is True
        assert retriever._score_threshold is None
        assert retriever._group_by is None
        assert retriever._group_size is None

    def test_from_dict_no_filter_policy(self):
        data = {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "init_parameters": {"location": ":memory:", "index": "test"},
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                },
                "filters": None,
                "top_k": 5,
                "scale_score": False,
                "return_embedding": True,
                "score_threshold": None,
                "group_by": None,
                "group_size": None,
            },
        }
        retriever = QdrantSparseEmbeddingRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE
        assert retriever._scale_score is False
        assert retriever._return_embedding is True
        assert retriever._score_threshold is None
        assert retriever._group_by is None
        assert retriever._group_size is None

    def test_run(self):
        mock_store = Mock(spec=QdrantDocumentStore)
        sparse = SparseEmbedding(indices=[0, 5], values=[0.1, 0.7])
        mock_store._query_by_sparse.return_value = [Document(content="doc", sparse_embedding=sparse)]

        retriever = QdrantSparseEmbeddingRetriever(document_store=mock_store)
        res = retriever.run(query_sparse_embedding=sparse, top_k=4)

        call_kwargs = mock_store._query_by_sparse.call_args.kwargs
        assert call_kwargs["query_sparse_embedding"] == sparse
        assert call_kwargs["top_k"] == 4
        assert res["documents"][0].content == "doc"

    @pytest.mark.asyncio
    async def test_run_async(self):
        mock_store = Mock(spec=QdrantDocumentStore)
        sparse = SparseEmbedding(indices=[0, 5], values=[0.1, 0.7])
        mock_store._query_by_sparse_async = AsyncMock(return_value=[Document(content="doc", sparse_embedding=sparse)])

        retriever = QdrantSparseEmbeddingRetriever(document_store=mock_store)
        res = await retriever.run_async(query_sparse_embedding=sparse)

        mock_store._query_by_sparse_async.assert_awaited_once()
        assert res["documents"][0].content == "doc"

    def test_run_raises_when_merge_with_native_filter(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantSparseEmbeddingRetriever(
            document_store=document_store,
            filters=rest.Filter(must=[]),
            filter_policy=FilterPolicy.MERGE,
        )
        sparse = SparseEmbedding(indices=[0], values=[0.1])
        with pytest.raises(ValueError, match="Native Qdrant filters"):
            retriever.run(query_sparse_embedding=sparse)

    @pytest.mark.asyncio
    async def test_run_async_raises_when_merge_with_native_filter(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store, filter_policy=FilterPolicy.MERGE)
        sparse = SparseEmbedding(indices=[0], values=[0.1])
        with pytest.raises(ValueError, match="Native Qdrant filters"):
            await retriever.run_async(query_sparse_embedding=sparse, filters=rest.Filter(must=[]))


@pytest.mark.integration
class TestQdrantSparseEmbeddingRetrieverIntegration(FilterableDocsFixtureMixin):
    def test_run(self, filterable_docs: list[Document], generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        # Add fake sparse embedding to documents
        filterable_docs = [replace(doc, sparse_embedding=generate_sparse_embedding()) for doc in filterable_docs]

        document_store.write_documents(filterable_docs)
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])

        results: list[Document] = retriever.run(query_sparse_embedding=sparse_embedding)["documents"]
        assert len(results) == 10

        results = retriever.run(query_sparse_embedding=sparse_embedding, top_k=5, return_embedding=True)["documents"]
        assert len(results) == 5

        for document in results:
            assert document.sparse_embedding

    def test_run_with_group_by(self, filterable_docs: list[Document], generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        # Add fake sparse embedding to documents
        filterable_docs = [
            replace(
                doc,
                sparse_embedding=generate_sparse_embedding(),
                meta={"group_field": index // 2},  # So at least two docs have same group each time
            )
            for index, doc in enumerate(filterable_docs)
        ]
        document_store.write_documents(filterable_docs)
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        results = retriever.run(
            query_sparse_embedding=sparse_embedding,
            top_k=3,
            return_embedding=True,
            group_by="meta.group_field",
            group_size=2,
        )["documents"]
        assert len(results) >= 3  # This test is Flaky
        assert len(results) <= 6  # This test is Flaky

        for document in results:
            assert document.sparse_embedding

    @pytest.mark.asyncio
    async def test_run_async(self, filterable_docs: list[Document], generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        # Add fake sparse embedding to documents
        filterable_docs = [replace(doc, sparse_embedding=generate_sparse_embedding()) for doc in filterable_docs]

        await document_store.write_documents_async(filterable_docs)
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])

        result = await retriever.run_async(query_sparse_embedding=sparse_embedding)
        assert len(result["documents"]) == 10

        result = await retriever.run_async(query_sparse_embedding=sparse_embedding, top_k=5, return_embedding=True)
        assert len(result["documents"]) == 5

        for document in result["documents"]:
            assert document.sparse_embedding

    @pytest.mark.asyncio
    async def test_run_with_group_by_async(self, filterable_docs: list[Document], generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        # Add fake sparse embedding to documents
        filterable_docs = [
            replace(
                doc,
                sparse_embedding=generate_sparse_embedding(),
                meta={"group_field": index // 2},  # So at least two docs have same group each time
            )
            for index, doc in enumerate(filterable_docs)
        ]
        await document_store.write_documents_async(filterable_docs)
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        result = await retriever.run_async(
            query_sparse_embedding=sparse_embedding,
            top_k=3,
            return_embedding=True,
            group_by="meta.group_field",
            group_size=2,
        )
        assert len(result["documents"]) >= 3  # This test is Flaky
        assert len(result["documents"]) <= 6  # This test is Flaky

        for document in result["documents"]:
            assert document.sparse_embedding
