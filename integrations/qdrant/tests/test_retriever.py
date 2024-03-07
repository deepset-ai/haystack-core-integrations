from typing import List

from haystack.dataclasses import Document
from haystack.testing.document_store import (
    FilterableDocsFixtureMixin,
    _random_embeddings,
)
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class TestQdrantRetriever(FilterableDocsFixtureMixin):
    def test_init_default(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
        retriever = QdrantEmbeddingRetriever(document_store=document_store)
        assert retriever._document_store == document_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._return_embedding is False

    def test_to_dict(self):
        document_store = QdrantDocumentStore(location=":memory:", index="test")
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
        document_store = QdrantDocumentStore(location=":memory:", index="Boi")

        document_store.write_documents(filterable_docs)

        retriever = QdrantEmbeddingRetriever(document_store=document_store)

        results: List[Document] = retriever.run(query_embedding=_random_embeddings(768))

        assert len(results["documents"]) == 10  # type: ignore

        results = retriever.run(query_embedding=_random_embeddings(768), top_k=5, return_embedding=False)

        assert len(results["documents"]) == 5  # type: ignore

        for document in results["documents"]:  # type: ignore
            assert document.embedding is None
