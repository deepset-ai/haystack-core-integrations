from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def test_to_dict():
    document_store = QdrantDocumentStore(location=":memory:", index="test")

    expected = {
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
    }

    assert document_store.to_dict() == expected


def test_from_dict():
    document_store = QdrantDocumentStore.from_dict(
        {
            "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "location": ":memory:",
                "index": "test",
                "embedding_dim": 768,
                "on_disk": False,
                "force_disable_check_same_thread": False,
                "use_sparse_embeddings": True,
                "sparse_idf": True,
                "similarity": "cosine",
                "return_embedding": False,
                "progress_bar": True,
                "recreate_index": True,
                "shard_number": None,
                "quantization_config": None,
                "init_from": None,
                "wait_result_from_api": True,
                "metadata": {},
                "write_batch_size": 1000,
                "scroll_size": 10000,
                "payload_fields_to_index": None,
            },
        }
    )

    assert all(
        [
            document_store.index == "test",
            document_store.force_disable_check_same_thread is False,
            document_store.use_sparse_embeddings is True,
            document_store.sparse_idf is True,
            document_store.on_disk is False,
            document_store.similarity == "cosine",
            document_store.return_embedding is False,
            document_store.progress_bar,
            document_store.recreate_index is True,
            document_store.shard_number is None,
            document_store.replication_factor is None,
            document_store.write_consistency_factor is None,
            document_store.on_disk_payload is None,
            document_store.hnsw_config is None,
            document_store.optimizers_config is None,
            document_store.wal_config is None,
            document_store.quantization_config is None,
            document_store.init_from is None,
            document_store.wait_result_from_api,
            document_store.metadata == {},
            document_store.write_batch_size == 1000,
            document_store.scroll_size == 10000,
            document_store.api_key == Secret.from_env_var("ENV_VAR", strict=False),
            document_store.payload_fields_to_index is None,
        ]
    )
