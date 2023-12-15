from qdrant_haystack import QdrantDocumentStore


def test_to_dict():
    document_store = QdrantDocumentStore(location=":memory:", index="test")

    expected = {
        "type": "qdrant_haystack.document_store.QdrantDocumentStore",
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
        },
    }

    assert document_store.to_dict() == expected


def test_from_dict():
    document_store = QdrantDocumentStore.from_dict(
        {
            "type": "qdrant_haystack.document_store.QdrantDocumentStore",
            "init_parameters": {
                "location": ":memory:",
                "index": "test",
                "embedding_dim": 768,
                "content_field": "content",
                "name_field": "name",
                "embedding_field": "embedding",
                "similarity": "cosine",
                "return_embedding": False,
                "progress_bar": True,
                "duplicate_documents": "overwrite",
                "recreate_index": True,
                "shard_number": None,
                "quantization_config": None,
                "init_from": None,
                "wait_result_from_api": True,
                "metadata": {},
                "write_batch_size": 1000,
                "scroll_size": 10000,
            },
        }
    )

    assert all(
        [
            document_store.index == "test",
            document_store.content_field == "content",
            document_store.name_field == "name",
            document_store.embedding_field == "embedding",
            document_store.similarity == "cosine",
            document_store.return_embedding is False,
            document_store.progress_bar,
            document_store.duplicate_documents == "overwrite",
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
        ]
    )
