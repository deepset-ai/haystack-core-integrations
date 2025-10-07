from typing import List
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    WriteDocumentsTest,
    _random_embeddings,
)
from haystack.utils import Secret
from qdrant_client.http import models as rest

from haystack_integrations.document_stores.qdrant.document_store import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    QdrantDocumentStore,
    QdrantStoreError,
)


class TestQdrantDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=False,
        )

    def test_init_is_lazy(self):
        with patch("haystack_integrations.document_stores.qdrant.document_store.qdrant_client") as mocked_qdrant:
            QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
            mocked_qdrant.assert_not_called()

    def test_prepare_client_params_no_mutability(self):
        metadata = {"key": "value"}
        doc_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=False,
            metadata=metadata,
        )
        client_params = doc_store._prepare_client_params()
        # Mutate value of metadata in client_params
        client_params["metadata"] = client_params["metadata"].update({"new_key": "new_value"})

        # Assert that the original metadata in the document store is unchanged
        assert metadata == {"key": "value"}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        doc_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=False,
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
        )
        expected_dict = {
            "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
            "init_parameters": {
                "location": ":memory:",
                "url": None,
                "port": 6333,
                "grpc_port": 6334,
                "prefer_grpc": False,
                "https": None,
                "api_key": {
                    "env_vars": ["QDRANT_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "prefix": None,
                "timeout": None,
                "host": None,
                "path": None,
                "force_disable_check_same_thread": False,
                "index": "Document",
                "embedding_dim": 768,
                "on_disk": False,
                "use_sparse_embeddings": False,
                "sparse_idf": False,
                "similarity": "cosine",
                "return_embedding": True,
                "progress_bar": True,
                "recreate_index": True,
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
        assert doc_store.to_dict() == expected_dict

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test.
        """

        # Check that the lengths of the lists are the same
        assert len(received) == len(expected)

        # Check that the sets are equal, meaning the content and IDs match regardless of order
        assert {doc.id for doc in received} == {doc.id for doc in expected}

    def test_write_documents(self, document_store: QdrantDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_sparse_configuration(self):
        document_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            use_sparse_embeddings=True,
            sparse_idf=True,
        )
        document_store._initialize_client()

        sparse_config = document_store._client.get_collection("Document").config.params.sparse_vectors

        assert SPARSE_VECTORS_NAME in sparse_config

        # check that the `sparse_idf` parameter takes effect
        assert hasattr(sparse_config[SPARSE_VECTORS_NAME], "modifier")
        assert sparse_config[SPARSE_VECTORS_NAME].modifier == rest.Modifier.IDF

    def test_query_hybrid(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)

        docs = []
        for i in range(20):
            docs.append(
                Document(
                    content=f"doc {i}", sparse_embedding=generate_sparse_embedding(), embedding=_random_embeddings(768)
                )
            )

        document_store.write_documents(docs)

        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        results: List[Document] = document_store._query_hybrid(
            query_sparse_embedding=sparse_embedding, query_embedding=embedding, top_k=10, return_embedding=True
        )
        assert len(results) == 10

        for document in results:
            assert document.sparse_embedding
            assert document.embedding

    def test_query_hybrid_with_group_by(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)

        docs = []
        for i in range(20):
            docs.append(
                Document(
                    content=f"doc {i}",
                    sparse_embedding=generate_sparse_embedding(),
                    embedding=_random_embeddings(768),
                    meta={"group_field": i // 2},
                )
            )

        document_store.write_documents(docs)

        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        results: List[Document] = document_store._query_hybrid(
            query_sparse_embedding=sparse_embedding,
            query_embedding=embedding,
            top_k=3,
            return_embedding=True,
            group_by="meta.group_field",
            group_size=2,
        )
        assert len(results) == 6

        for document in results:
            assert document.sparse_embedding
            assert document.embedding

    def test_query_hybrid_fail_without_sparse_embedding(self, document_store):
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        with pytest.raises(QdrantStoreError):
            document_store._query_hybrid(
                query_sparse_embedding=sparse_embedding,
                query_embedding=embedding,
            )

    def test_query_hybrid_search_batch_failure(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        document_store._initialize_client()
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        with patch.object(document_store._client, "query_points", side_effect=Exception("query_points")):
            with pytest.raises(QdrantStoreError):
                document_store._query_hybrid(query_sparse_embedding=sparse_embedding, query_embedding=embedding)

    def test_set_up_collection_with_existing_incompatible_collection(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        document_store._initialize_client()
        # Mock collection info with named vectors but missing DENSE_VECTORS_NAME
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = {"some_other_vector": MagicMock()}

        with patch.object(document_store._client, "collection_exists", return_value=True), patch.object(
            document_store._client, "get_collection", return_value=mock_collection_info
        ):
            with pytest.raises(QdrantStoreError, match="created outside of Haystack"):
                document_store._set_up_collection("test_collection", 768, False, "cosine", True, False)

    def test_set_up_collection_use_sparse_embeddings_true_without_named_vectors(self):
        """Test that an error is raised when use_sparse_embeddings is True but collection doesn't have named vectors"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        document_store._initialize_client()

        # Mock collection info without named vectors
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = MagicMock(spec=rest.VectorsConfig)

        with patch.object(document_store._client, "collection_exists", return_value=True), patch.object(
            document_store._client, "get_collection", return_value=mock_collection_info
        ):
            with pytest.raises(QdrantStoreError, match="without sparse embedding vectors"):
                document_store._set_up_collection("test_collection", 768, False, "cosine", True, False)

    def test_set_up_collection_use_sparse_embeddings_false_with_named_vectors(self):
        """Test that an error is raised when use_sparse_embeddings is False but collection has named vectors"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False)
        document_store._initialize_client()
        # Mock collection info with named vectors
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = {DENSE_VECTORS_NAME: MagicMock()}

        with patch.object(document_store._client, "collection_exists", return_value=True), patch.object(
            document_store._client, "get_collection", return_value=mock_collection_info
        ):
            with pytest.raises(QdrantStoreError, match="with sparse embedding vectors"):
                document_store._set_up_collection("test_collection", 768, False, "cosine", False, False)

    def test_set_up_collection_with_distance_mismatch(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False, similarity="cosine")
        document_store._initialize_client()

        # Mock collection info with different distance
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = MagicMock()
        mock_collection_info.config.params.vectors.distance = rest.Distance.DOT
        mock_collection_info.config.params.vectors.size = 768

        with patch.object(document_store._client, "collection_exists", return_value=True), patch.object(
            document_store._client, "get_collection", return_value=mock_collection_info
        ):
            with pytest.raises(ValueError, match="different similarity"):
                document_store._set_up_collection("test_collection", 768, False, "cosine", False, False)

    def test_set_up_collection_with_dimension_mismatch(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False, similarity="cosine")
        document_store._initialize_client()
        # Mock collection info with different vector size
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = MagicMock()
        mock_collection_info.config.params.vectors.distance = rest.Distance.COSINE
        mock_collection_info.config.params.vectors.size = 512

        with patch.object(document_store._client, "collection_exists", return_value=True), patch.object(
            document_store._client, "get_collection", return_value=mock_collection_info
        ):
            with pytest.raises(ValueError, match="different vector size"):
                document_store._set_up_collection("test_collection", 768, False, "cosine", False, False)

    def test_delete_all_documents_no_index_recreation(self, document_store):
        document_store._initialize_client()

        # write some documents
        docs = [Document(id=str(i)) for i in range(5)]
        document_store.write_documents(docs)

        # delete all documents without recreating the index
        document_store.delete_all_documents(recreate_index=False)
        assert document_store.count_documents() == 0

        # ensure the collection still exists by writing documents again
        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

    def test_delete_all_documents_index_recreation(self, document_store):
        document_store._initialize_client()

        # write some documents
        docs = [Document(id=str(i)) for i in range(5)]
        document_store.write_documents(docs)

        # get the current document_store config
        config_before = document_store._client.get_collection(document_store.index)

        # delete all documents with recreating the index
        document_store.delete_all_documents(recreate_index=True)
        assert document_store.count_documents() == 0

        # assure that with the same config
        config_after = document_store._client.get_collection(document_store.index)

        assert config_before.config == config_after.config

        # ensure the collection still exists by writing documents again
        document_store.write_documents(docs)
        assert document_store.count_documents() == 5
