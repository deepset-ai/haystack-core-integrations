from typing import List
from unittest.mock import Mock

from haystack.dataclasses import Document, SparseEmbedding
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
        assert retriever._return_embedding is False

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
                        "content_field": "content",
                        "name_field": "name",
                        "embedding_field": "embedding",
                        "use_sparse_embeddings": False,
                        "similarity": "cosine",
                        "return_embedding": False,
                        "progress_bar": True,
                        "duplicate_documents": "overwrite",
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
                "scale_score": True,
                "return_embedding": False,
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
                "scale_score": False,
                "return_embedding": True,
            },
        }
        retriever = QdrantEmbeddingRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._scale_score is False
        assert retriever._return_embedding is True

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
        assert retriever._return_embedding is False

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
                        "content_field": "content",
                        "name_field": "name",
                        "embedding_field": "embedding",
                        "use_sparse_embeddings": False,
                        "similarity": "cosine",
                        "return_embedding": False,
                        "progress_bar": True,
                        "duplicate_documents": "overwrite",
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
                "scale_score": True,
                "return_embedding": False,
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
            },
        }
        retriever = QdrantSparseEmbeddingRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._scale_score is False
        assert retriever._return_embedding is True

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
        assert retriever._return_embedding is False

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
                        "content_field": "content",
                        "name_field": "name",
                        "embedding_field": "embedding",
                        "use_sparse_embeddings": False,
                        "similarity": "cosine",
                        "return_embedding": False,
                        "progress_bar": True,
                        "duplicate_documents": "overwrite",
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
                "return_embedding": True,
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
                "return_embedding": True,
            },
        }
        retriever = QdrantHybridRetriever.from_dict(data)
        assert isinstance(retriever._document_store, QdrantDocumentStore)
        assert retriever._document_store.index == "test"
        assert retriever._filters is None
        assert retriever._top_k == 5
        assert retriever._return_embedding

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
