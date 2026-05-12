from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.testing.document_store import (
    FilterableDocsFixtureMixin,
    _random_embeddings,
)
from qdrant_client.http import models as rest

from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class TestQdrantRetriever:
    def test_init_raises_when_document_store_is_not_qdrant(self):
        with pytest.raises(ValueError, match="must be an instance of QdrantDocumentStore"):
            QdrantEmbeddingRetriever(document_store="not a document store")

    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test", use_sparse_embeddings=False)
        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        assert retriever._document_store == document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._return_embedding is False
        assert retriever._score_threshold is None
        assert retriever._group_by is None
        assert retriever._group_size is None

        retriever = QdrantEmbeddingRetriever(document_store=document_store, filter_policy="replace")
        assert retriever._filter_policy == FilterPolicy.REPLACE

        with pytest.raises(ValueError):
            QdrantEmbeddingRetriever(document_store=document_store, filter_policy="invalid")

    def test_to_dict(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test", use_sparse_embeddings=False)
        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        res = retriever.to_dict()
        assert res == {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever",
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
                "filter_policy": "replace",
                "scale_score": False,
                "return_embedding": False,
                "score_threshold": None,
                "group_by": None,
                "group_size": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "init_parameters": {"location": ":memory:", "index": "test"},
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                },
                "filters": None,
                "top_k": 5,
                "filter_policy": "replace",
                "scale_score": False,
                "return_embedding": True,
                "score_threshold": None,
                "group_by": None,
                "group_size": None,
            },
        }
        retriever = QdrantEmbeddingRetriever.from_dict(data)
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
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever",
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
        retriever = QdrantEmbeddingRetriever.from_dict(data)
        assert retriever._filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE

    def test_run(self):
        mock_store = Mock(spec=QdrantDocumentStore)
        mock_store._query_by_embedding.return_value = [Document(content="doc", embedding=[0.1, 0.2])]

        retriever = QdrantEmbeddingRetriever(
            document_store=mock_store,
            filters={"field": "meta.name", "operator": "==", "value": "foo"},
            top_k=7,
            scale_score=True,
            return_embedding=True,
            score_threshold=0.2,
        )
        res = retriever.run(query_embedding=[0.5, 0.7], top_k=3)

        call_kwargs = mock_store._query_by_embedding.call_args.kwargs
        assert call_kwargs["query_embedding"] == [0.5, 0.7]
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["scale_score"] is True
        assert call_kwargs["return_embedding"] is True
        assert call_kwargs["score_threshold"] == 0.2
        assert call_kwargs["filters"] == {"field": "meta.name", "operator": "==", "value": "foo"}
        assert res["documents"][0].content == "doc"

    @pytest.mark.asyncio
    async def test_run_async(self):
        mock_store = Mock(spec=QdrantDocumentStore)
        mock_store._query_by_embedding_async = AsyncMock(return_value=[Document(content="doc", embedding=[0.1])])

        retriever = QdrantEmbeddingRetriever(document_store=mock_store)
        res = await retriever.run_async(query_embedding=[0.5])

        mock_store._query_by_embedding_async.assert_awaited_once()
        assert res["documents"][0].content == "doc"

    def test_run_raises_when_merge_with_native_init_filter(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            filters=rest.Filter(must=[]),
            filter_policy=FilterPolicy.MERGE,
        )
        with pytest.raises(ValueError, match="Native Qdrant filters"):
            retriever.run(query_embedding=[0.1])

    def test_run_raises_when_merge_with_native_runtime_filter(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantEmbeddingRetriever(document_store=document_store, filter_policy=FilterPolicy.MERGE)
        with pytest.raises(ValueError, match="Native Qdrant filters"):
            retriever.run(query_embedding=[0.1], filters=rest.Filter(must=[]))

    @pytest.mark.asyncio
    async def test_run_async_raises_when_merge_with_native_filter(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            filters=rest.Filter(must=[]),
            filter_policy=FilterPolicy.MERGE,
        )
        with pytest.raises(ValueError, match="Native Qdrant filters"):
            await retriever.run_async(query_embedding=[0.1])


@pytest.mark.integration
class TestQdrantEmbeddingRetrieverIntegration(FilterableDocsFixtureMixin):
    def test_run(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=False)

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)

        results: list[Document] = retriever.run(query_embedding=_random_embeddings(768))["documents"]
        assert len(results) == 10

        results = retriever.run(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)["documents"]
        assert len(results) == 5

        for document in results:
            assert document.embedding is None

    def test_run_filters(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=False)

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.name", "operator": "==", "value": "name_0"},
            filter_policy=FilterPolicy.MERGE,
        )

        results: list[Document] = retriever.run(query_embedding=_random_embeddings(768))["documents"]
        assert len(results) == 3

        results = retriever.run(
            query_embedding=_random_embeddings(768),
            top_k=5,
            filters={"field": "meta.chapter", "operator": "==", "value": "abstract"},
            return_embedding=False,
        )["documents"]
        # we need to combine init filter and run filter as the policy is MERGE
        # when we combine these filters we use AND logical operator by default
        # so the result should be 1 as we have only one document that matches both filters
        assert len(results) == 1

        for document in results:
            assert document.embedding is None

    def test_run_with_score_threshold(self):
        document_store = QdrantDocumentStore(
            embedding_dim=4, location=":memory:", similarity="cosine", index="Boi", use_sparse_embeddings=False
        )
        document_store._initialize_client()
        document_store.write_documents(
            [
                Document(
                    content="Yet another document",
                    embedding=[-0.1, -0.9, -10.0, -0.2],
                ),
                Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.5, 1.0]),
            ]
        )

        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        results = retriever.run(
            query_embedding=[0.9, 0.9, 0.9, 0.9], top_k=5, return_embedding=False, score_threshold=0.5
        )["documents"]
        assert len(results) == 2

    def test_run_with_sparse_activated(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)

        results: list[Document] = retriever.run(query_embedding=_random_embeddings(768))["documents"]

        assert len(results) == 10

        results = retriever.run(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)["documents"]

        assert len(results) == 5

        for document in results:
            assert document.embedding is None

    def test_run_with_group_by(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)
        # Add group_field metadata to documents
        filterable_docs = [
            replace(doc, meta={"group_field": index // 2})  # So at least two docs have same group each time
            for index, doc in enumerate(filterable_docs)
        ]
        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        results = retriever.run(
            query_embedding=_random_embeddings(768),
            top_k=3,
            return_embedding=False,
            group_by="meta.group_field",
            group_size=2,
        )["documents"]
        assert len(results) >= 3  # This test is Flaky
        assert len(results) <= 6  # This test is Flaky
        for document in results:
            assert document.embedding is None

    @pytest.mark.asyncio
    async def test_run_async(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=False)

        await document_store.write_documents_async(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        result = await retriever.run_async(query_embedding=_random_embeddings(768))
        assert len(result["documents"]) == 10

        result = await retriever.run_async(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)
        assert len(result["documents"]) == 5

        for document in result["documents"]:
            assert document.embedding is None

    @pytest.mark.asyncio
    async def test_run_filters_async(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=False)

        await document_store.write_documents_async(filterable_docs)

        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.name", "operator": "==", "value": "name_0"},
            filter_policy=FilterPolicy.MERGE,
        )

        results = await retriever.run_async(query_embedding=_random_embeddings(768))
        assert len(results["documents"]) == 3

        result = await retriever.run_async(
            query_embedding=_random_embeddings(768),
            top_k=5,
            filters={"field": "meta.chapter", "operator": "==", "value": "abstract"},
            return_embedding=False,
        )
        assert len(result["documents"]) == 1
        # we need to combine init filter and run filter as the policy is MERGE
        # when we combine these filters we use AND logical operator by default
        # so the result should be 1 as we have only one document that matches both filters

        for document in result["documents"]:
            assert document.embedding is None

    @pytest.mark.asyncio
    async def test_run_with_score_threshold_async(self):
        document_store = QdrantDocumentStore(
            embedding_dim=4, location=":memory:", similarity="cosine", index="Boi", use_sparse_embeddings=False
        )
        await document_store.write_documents_async(
            [
                Document(
                    content="Yet another document",
                    embedding=[-0.1, -0.9, -10.0, -0.2],
                ),
                Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.5, 1.0]),
            ]
        )

        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        result = await retriever.run_async(
            query_embedding=[0.9, 0.9, 0.9, 0.9], top_k=5, return_embedding=False, score_threshold=0.5
        )
        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_run_with_sparse_activated_async(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        await document_store.write_documents_async(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)

        result = await retriever.run_async(query_embedding=_random_embeddings(768))

        assert len(result["documents"]) == 10

        result = await retriever.run_async(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)

        assert len(result["documents"]) == 5

        for document in result["documents"]:
            assert document.embedding is None

    @pytest.mark.asyncio
    async def test_run_with_group_by_async(self, filterable_docs: list[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)
        # Add group_field metadata to documents
        filterable_docs = [
            replace(doc, meta={"group_field": index // 2})  # So at least two docs have same group each time
            for index, doc in enumerate(filterable_docs)
        ]
        await document_store.write_documents_async(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        result = await retriever.run_async(
            query_embedding=_random_embeddings(768),
            top_k=3,
            return_embedding=False,
            group_by="meta.group_field",
            group_size=2,
        )
        assert len(result["documents"]) >= 3  # This test is Flaky
        assert len(result["documents"]) <= 6  # This test is Flaky
        for document in result["documents"]:
            assert document.embedding is None
