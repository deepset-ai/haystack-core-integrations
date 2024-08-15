from typing import List
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from haystack.testing.document_store import (
    FilterableDocsFixtureMixin,
    _random_embeddings,
)
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantHybridRetriever,
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class TestQdrantRetriever(FilterableDocsFixtureMixin):
    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test", use_sparse_embeddings=False)
        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        assert retriever._document_store == document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._return_embedding is False
        assert retriever._score_threshold is None

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
                        "init_from": None,
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

    def test_run(self, filterable_docs: List[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=False)

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)

        results: List[Document] = retriever.run(query_embedding=_random_embeddings(768))["documents"]
        assert len(results) == 10

        results = retriever.run(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)["documents"]
        assert len(results) == 5

        for document in results:
            assert document.embedding is None

    def test_run_filters(self, filterable_docs: List[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=False)

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.name", "operator": "==", "value": "name_0"},
            filter_policy=FilterPolicy.MERGE,
        )

        results: List[Document] = retriever.run(query_embedding=_random_embeddings(768))["documents"]
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

    def test_run_with_sparse_activated(self, filterable_docs: List[Document]):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)

        results: List[Document] = retriever.run(query_embedding=_random_embeddings(768))["documents"]

        assert len(results) == 10

        results = retriever.run(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)["documents"]

        assert len(results) == 5

        for document in results:
            assert document.embedding is None


class TestQdrantSparseEmbeddingRetriever(FilterableDocsFixtureMixin):
    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        assert retriever._document_store == document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._return_embedding is False
        assert retriever._score_threshold is None

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
                        "init_from": None,
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

    def test_run(self, filterable_docs: List[Document], generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", index="Boi", use_sparse_embeddings=True)

        # Add fake sparse embedding to documents
        for doc in filterable_docs:
            doc.sparse_embedding = generate_sparse_embedding()

        document_store.write_documents(filterable_docs)
        retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])

        results: List[Document] = retriever.run(query_sparse_embedding=sparse_embedding)["documents"]
        assert len(results) == 10

        results = retriever.run(query_sparse_embedding=sparse_embedding, top_k=5, return_embedding=True)["documents"]
        assert len(results) == 5

        for document in results:
            assert document.sparse_embedding


class TestQdrantHybridRetriever:
    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test", use_sparse_embeddings=True)
        retriever = QdrantHybridRetriever(document_store=document_store)

        assert retriever._document_store == document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._return_embedding is False
        assert retriever._score_threshold is None

        retriever = QdrantHybridRetriever(document_store=document_store, filter_policy="replace")
        assert retriever._filter_policy == FilterPolicy.REPLACE

        with pytest.raises(ValueError):
            QdrantHybridRetriever(document_store=document_store, filter_policy="invalid")

    def test_to_dict(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantHybridRetriever(document_store=document_store, top_k=5, return_embedding=True)
        res = retriever.to_dict()
        assert res == {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever",
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
                        "init_from": None,
                        "wait_result_from_api": True,
                        "metadata": {},
                        "write_batch_size": 100,
                        "scroll_size": 10000,
                        "payload_fields_to_index": None,
                    },
                },
                "filters": None,
                "top_k": 5,
                "filter_policy": "replace",
                "return_embedding": True,
                "score_threshold": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever",
            "init_parameters": {
                "document_store": {
                    "init_parameters": {"location": ":memory:", "index": "test"},
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                },
                "filters": None,
                "top_k": 5,
                "filter_policy": "replace",
                "return_embedding": True,
                "score_threshold": None,
            },
        }
        retriever = QdrantHybridRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._return_embedding
        assert retriever._score_threshold is None

    def test_from_dict_no_filter_policy(self):
        data = {
            "type": "haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever",
            "init_parameters": {
                "document_store": {
                    "init_parameters": {"location": ":memory:", "index": "test"},
                    "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                },
                "filters": None,
                "top_k": 5,
                "return_embedding": True,
                "score_threshold": None,
            },
        }
        retriever = QdrantHybridRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE
        assert retriever._return_embedding
        assert retriever._score_threshold is None

    def test_run(self):
        mock_store = Mock(spec=QdrantDocumentStore)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        mock_store._query_hybrid.return_value = [
            Document(content="Test doc", embedding=[0.1, 0.2], sparse_embedding=sparse_embedding)
        ]

        retriever = QdrantHybridRetriever(document_store=mock_store)
        res = retriever.run(
            query_embedding=[0.5, 0.7], query_sparse_embedding=SparseEmbedding(indices=[0, 5], values=[0.1, 0.7])
        )

        call_args = mock_store._query_hybrid.call_args
        assert call_args[1]["query_embedding"] == [0.5, 0.7]
        assert call_args[1]["query_sparse_embedding"].indices == [0, 5]
        assert call_args[1]["query_sparse_embedding"].values == [0.1, 0.7]
        assert call_args[1]["top_k"] == 10
        assert call_args[1]["return_embedding"] is False

        assert res["documents"][0].content == "Test doc"
        assert res["documents"][0].embedding == [0.1, 0.2]
        assert res["documents"][0].sparse_embedding == sparse_embedding
