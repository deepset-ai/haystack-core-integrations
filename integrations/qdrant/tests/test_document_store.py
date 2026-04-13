from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
    _random_embeddings,
)
from haystack.utils import Secret
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from haystack_integrations.document_stores.qdrant.document_store import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    QdrantDocumentStore,
    QdrantStoreError,
)


class TestQdrantDocumentStoreUnit:
    def test_init_is_lazy(self):
        with patch("haystack_integrations.document_stores.qdrant.document_store.qdrant_client") as mocked_qdrant:
            QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
            mocked_qdrant.assert_not_called()

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
                "wait_result_from_api": True,
                "metadata": {},
                "write_batch_size": 100,
                "scroll_size": 10000,
                "payload_fields_to_index": None,
            },
        }
        assert doc_store.to_dict() == expected_dict

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

        with (
            patch.object(document_store._client, "collection_exists", return_value=True),
            patch.object(document_store._client, "get_collection", return_value=mock_collection_info),
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

        with (
            patch.object(document_store._client, "collection_exists", return_value=True),
            patch.object(document_store._client, "get_collection", return_value=mock_collection_info),
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

        with (
            patch.object(document_store._client, "collection_exists", return_value=True),
            patch.object(document_store._client, "get_collection", return_value=mock_collection_info),
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

        with (
            patch.object(document_store._client, "collection_exists", return_value=True),
            patch.object(document_store._client, "get_collection", return_value=mock_collection_info),
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

        with (
            patch.object(document_store._client, "collection_exists", return_value=True),
            patch.object(document_store._client, "get_collection", return_value=mock_collection_info),
        ):
            with pytest.raises(ValueError, match="different vector size"):
                document_store._set_up_collection("test_collection", 768, False, "cosine", False, False)

    def test_count_documents_on_unexpected_response(self):
        """Test count_documents when Qdrant raises UnexpectedResponse"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(
            document_store._client,
            "count",
            side_effect=UnexpectedResponse(404, "Not Found", b"Collection not found", {}),
        ):
            result = document_store.count_documents()
            assert result == 0

    def test_count_documents_on_value_error(self):
        """Test count_documents when Qdrant raises ValueError"""
        document_store = QdrantDocumentStore(location=":memory:")

        mock_client = MagicMock()
        mock_client.count.side_effect = ValueError("Invalid collection")

        with patch.object(document_store, "_initialize_client") as mock_init:
            mock_init.return_value = None
            document_store._client = mock_client

            result = document_store.count_documents()
            assert result == 0

    @pytest.mark.asyncio
    async def test_count_documents_async_on_unexpected_response(self):
        """Test count_documents_async when Qdrant raises UnexpectedResponse"""
        document_store = QdrantDocumentStore(location=":memory:")
        mock_client = MagicMock()
        mock_client.count.side_effect = UnexpectedResponse(404, "Server error", b"Server error", {})

        with patch.object(document_store, "_initialize_async_client") as mock_init:
            mock_init.return_value = None
            document_store._async_client = mock_client

            result = await document_store.count_documents_async()
            assert result == 0

        with patch.object(
            document_store._async_client,
            "count",
            side_effect=UnexpectedResponse(404, "Not Found", b"Collection not found", {}),
        ):
            result = await document_store.count_documents_async()
            assert result == 0

    @pytest.mark.asyncio
    async def test_count_documents_async_on_value_error(self):
        """Test count_documents_async when Qdrant raises ValueError"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(
            document_store._async_client,
            "count",
            side_effect=ValueError("Invalid collection"),
        ):
            result = await document_store.count_documents_async()
            assert result == 0

    def test_write_documents_validates_document_types(self):
        """Test write_documents validates that all elements are Document instances"""
        document_store = QdrantDocumentStore(location=":memory:")

        with pytest.raises(ValueError, match="expects a list of Documents but got an element of"):
            document_store.write_documents(["not a document"])

    def test_write_documents_empty_list(self):
        """Test write_documents with empty list returns 0 and logs warning"""
        document_store = QdrantDocumentStore(location=":memory:")

        with patch("haystack_integrations.document_stores.qdrant.document_store.logger") as mock_logger:
            result = document_store.write_documents([])
            assert result == 0
            mock_logger.warning.assert_called_once_with("Calling QdrantDocumentStore.write_documents() with empty list")

    @pytest.mark.asyncio
    async def test_write_documents_async_validates_document_types(self):
        """Test write_documents_async validates that all elements are Document instances"""
        document_store = QdrantDocumentStore(location=":memory:")

        with pytest.raises(ValueError, match=r"Documents but got an element of"):
            await document_store.write_documents_async(["not a document"])

    @pytest.mark.asyncio
    async def test_write_documents_async_empty_list(self):
        """Test write_documents_async with empty list returns 0 and logs warning"""
        document_store = QdrantDocumentStore(location=":memory:")

        with patch("haystack_integrations.document_stores.qdrant.document_store.logger") as mock_logger:
            result = await document_store.write_documents_async([])
            assert result == 0
            mock_logger.warning.assert_called_once_with(
                "Calling QdrantDocumentStore.write_documents_async() with empty list"
            )

    def test_delete_documents_handles_keyerror(self):
        """Test delete_documents logs warning when KeyError is raised"""
        document_store = QdrantDocumentStore(location=":memory:")
        mock_client = MagicMock()
        document_store._client = mock_client

        mock_client.delete.side_effect = KeyError("Point not found")
        with patch("haystack_integrations.document_stores.qdrant.document_store.logger") as mock_logger:
            document_store.delete_documents(["non-existing-id"])
            mock_logger.warning.assert_called_once_with(
                "Called QdrantDocumentStore.delete_documents() on a non-existing ID",
            )

    @pytest.mark.asyncio
    async def test_filter_documents_async_validates_filters(self):
        """Test filter_documents_async calls _validate_filters"""
        document_store = QdrantDocumentStore(location=":memory:")

        invalid_filters = {"invalid": "filters"}

        with patch.object(QdrantDocumentStore, "_validate_filters", side_effect=ValueError("Invalid filters")):
            with pytest.raises(ValueError, match="Invalid filters"):
                await document_store.filter_documents_async(invalid_filters)

    def test_delete_by_filter_handles_exception(self):
        """Test delete_by_filter exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        mock_client = MagicMock()
        document_store._client = mock_client

        with patch(
            "haystack_integrations.document_stores.qdrant.document_store.convert_filters_to_qdrant",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(QdrantStoreError, match="Failed to delete documents by filter from Qdrant"):
                document_store.delete_by_filter({"field": "test", "operator": "==", "value": "test"})

    @pytest.mark.asyncio
    async def test_delete_by_filter_async_handles_exception(self):
        """Test delete_by_filter_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch(
            "haystack_integrations.document_stores.qdrant.document_store.convert_filters_to_qdrant",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(QdrantStoreError, match="Failed to delete documents by filter from Qdrant"):
                await document_store.delete_by_filter_async({"field": "test", "operator": "==", "value": "test"})

    def test_update_by_filter_handles_exception(self):
        """Test update_by_filter exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch(
            "haystack_integrations.document_stores.qdrant.document_store.convert_filters_to_qdrant",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(QdrantStoreError, match="Failed to update documents by filter in Qdrant"):
                document_store.update_by_filter({"field": "test", "operator": "==", "value": "test"}, {"new": "meta"})

    @pytest.mark.asyncio
    async def test_update_by_filter_async_handles_exception(self):
        """Test update_by_filter_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch(
            "haystack_integrations.document_stores.qdrant.document_store.convert_filters_to_qdrant",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(QdrantStoreError, match="Failed to update documents by filter in Qdrant"):
                await document_store.update_by_filter_async(
                    {"field": "test", "operator": "==", "value": "test"}, {"new": "meta"}
                )

    def test_count_documents_by_filter_handles_exception(self):
        """Test count_documents_by_filter exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "count", side_effect=ValueError("Invalid collection")):
            result = document_store.count_documents_by_filter({"field": "test", "operator": "==", "value": "test"})
            assert result == 0

    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async_handles_exception(self):
        """Test count_documents_by_filter_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "count", side_effect=ValueError("Invalid collection")):
            result = await document_store.count_documents_by_filter_async(
                {"field": "test", "operator": "==", "value": "test"}
            )
            assert result == 0

    def test_get_metadata_fields_info_handles_exception(self):
        """Test get_metadata_fields_info exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "get_collection", side_effect=ValueError("Collection missing")):
            result = document_store.get_metadata_fields_info()
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_metadata_fields_info_async_handles_exception(self):
        """Test get_metadata_fields_info_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "get_collection", side_effect=ValueError("Collection missing")):
            result = await document_store.get_metadata_fields_info_async()
            assert result == {}

    def test_get_metadata_field_min_max_handles_exception(self):
        """Test get_metadata_field_min_max exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "scroll", side_effect=Exception("Scroll failed")):
            result = document_store.get_metadata_field_min_max("test_field")
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_metadata_field_min_max_async_handles_exception(self):
        """Test get_metadata_field_min_max_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "scroll", side_effect=Exception("Scroll failed")):
            result = await document_store.get_metadata_field_min_max_async("test_field")
            assert result == {}

    def test_count_unique_metadata_by_filter_handles_exception(self):
        """Test count_unique_metadata_by_filter exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "scroll", side_effect=Exception("Scroll failed")):
            result = document_store.count_unique_metadata_by_filter(
                {"field": "test", "operator": "==", "value": "test"}, ["field"]
            )
            assert result == {"field": 0}

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_handles_exception(self):
        """Test count_unique_metadata_by_filter_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "scroll", side_effect=Exception("Scroll failed")):
            result = await document_store.count_unique_metadata_by_filter_async(
                {"field": "test", "operator": "==", "value": "test"}, ["field"]
            )
            assert result == {"field": 0}

    def test_get_metadata_field_unique_values_handles_exception(self):
        """Test get_metadata_field_unique_values exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "scroll", side_effect=Exception("Scroll failed")):
            result = document_store.get_metadata_field_unique_values("test_field")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_handles_exception(self):
        """Test get_metadata_field_unique_values_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "scroll", side_effect=Exception("Scroll failed")):
            result = await document_store.get_metadata_field_unique_values_async("test_field")
            assert result == []

    def test_delete_all_documents_handles_exception(self):
        """Test delete_all_documents exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "delete", side_effect=Exception("Delete failed")):
            with patch("haystack_integrations.document_stores.qdrant.document_store.logger") as mock_logger:
                document_store.delete_all_documents()
                mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_handles_exception(self):
        """Test delete_all_documents_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "delete", side_effect=Exception("Delete failed")):
            with patch("haystack_integrations.document_stores.qdrant.document_store.logger") as mock_logger:
                await document_store.delete_all_documents_async()
                mock_logger.warning.assert_called_once()

    def test_get_documents_by_id_handles_exception(self):
        """Test get_documents_by_id exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "retrieve", side_effect=Exception("Retrieve failed")):
            with pytest.raises(Exception, match="Retrieve failed"):
                document_store.get_documents_by_id(["test_id"])

    @pytest.mark.asyncio
    async def test_get_documents_by_id_async_handles_exception(self):
        """Test get_documents_by_id_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "retrieve", side_effect=Exception("Retrieve failed")):
            with pytest.raises(Exception, match="Retrieve failed"):
                await document_store.get_documents_by_id_async(["test_id"])

    def test__query_by_sparse_handles_exception(self):
        """Test _query_by_sparse exception handling"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        document_store._initialize_client()

        sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.1, 0.2])

        with patch.object(document_store._client, "query_points", side_effect=Exception("Query failed")):
            with pytest.raises(Exception, match="Query failed"):
                document_store._query_by_sparse(sparse_embedding)

    @pytest.mark.asyncio
    async def test__query_by_sparse_async_handles_exception(self):
        """Test _query_by_sparse_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        await document_store._initialize_async_client()

        sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.1, 0.2])

        with patch.object(document_store._async_client, "query_points", side_effect=Exception("Query failed")):
            with pytest.raises(Exception, match="Query failed"):
                await document_store._query_by_sparse_async(sparse_embedding)

    def test__query_by_embedding_handles_exception(self):
        """Test _query_by_embedding exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        embedding = [0.1] * 768

        with patch.object(document_store._client, "query_points", side_effect=Exception("Query failed")):
            with pytest.raises(Exception, match="Query failed"):
                document_store._query_by_embedding(embedding)

    @pytest.mark.asyncio
    async def test__query_by_embedding_async_handles_exception(self):
        """Test _query_by_embedding_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        embedding = [0.1] * 768

        with patch.object(document_store._async_client, "query_points", side_effect=Exception("Query failed")):
            with pytest.raises(Exception, match="Query failed"):
                await document_store._query_by_embedding_async(embedding)

    def test__query_hybrid_handles_exception(self):
        """Test _query_hybrid exception handling"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        document_store._initialize_client()

        embedding = [0.1] * 768
        sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.1, 0.2])

        with patch.object(document_store._client, "query_points", side_effect=Exception("Query failed")):
            with pytest.raises(QdrantStoreError, match="Error during hybrid search"):
                document_store._query_hybrid(embedding, sparse_embedding)

    @pytest.mark.asyncio
    async def test__query_hybrid_async_handles_exception(self):
        """Test _query_hybrid_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        await document_store._initialize_async_client()

        embedding = [0.1] * 768
        sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.1, 0.2])

        with patch.object(document_store._async_client, "query_points", side_effect=Exception("Query failed")):
            with pytest.raises(QdrantStoreError, match="Error during hybrid search"):
                await document_store._query_hybrid_async(embedding, sparse_embedding)

    def test__get_documents_generator_handles_exception(self):
        """Test _get_documents_generator exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()

        with patch.object(document_store._client, "scroll", side_effect=Exception("Scroll failed")):
            with pytest.raises(Exception, match="Scroll failed"):
                list(document_store._get_documents_generator())

    @pytest.mark.asyncio
    async def test__get_documents_generator_async_handles_exception(self):
        """Test _get_documents_generator_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()

        with patch.object(document_store._async_client, "scroll", side_effect=Exception("Scroll failed")):
            with pytest.raises(Exception, match="Scroll failed"):
                async for _doc in document_store._get_documents_generator_async():
                    pass

    def test__set_up_collection_handles_exception(self):
        """Test _set_up_collection exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._client = MagicMock()

        with (
            patch.object(document_store._client, "collection_exists", return_value=True),
            patch.object(document_store._client, "get_collection", side_effect=Exception("Get collection failed")),
        ):
            with pytest.raises(Exception, match="Get collection failed"):
                document_store._set_up_collection("test", 768, False, "cosine", False, False)

    @pytest.mark.asyncio
    async def test__set_up_collection_async_handles_exception(self):
        """Test _set_up_collection_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._async_client = MagicMock()

        document_store._async_client.collection_exists = AsyncMock(return_value=True)
        document_store._async_client.get_collection = AsyncMock(side_effect=Exception("Get collection failed"))

        with pytest.raises(Exception, match="Get collection failed"):
            await document_store._set_up_collection_async("test", 768, False, "cosine", False, False)

    def test_recreate_collection_handles_exception(self):
        """Test recreate_collection exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._client = MagicMock()
        document_store._client.collection_exists.return_value = True

        with patch.object(document_store._client, "delete_collection", side_effect=Exception("Delete failed")):
            with pytest.raises(Exception, match="Delete failed"):
                document_store.recreate_collection("test", rest.Distance.COSINE, 768)

    @pytest.mark.asyncio
    async def test_recreate_collection_async_handles_exception(self):
        """Test recreate_collection_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._async_client = MagicMock()
        document_store._async_client.collection_exists = AsyncMock(return_value=True)
        document_store._async_client.delete_collection = AsyncMock(side_effect=Exception("Delete failed"))

        with pytest.raises(Exception, match="Delete failed"):
            await document_store.recreate_collection_async("test", rest.Distance.COSINE, 768)

    def test__create_payload_index_handles_exception(self):
        """Test _create_payload_index exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._client = MagicMock()

        with patch.object(
            document_store._client, "create_payload_index", side_effect=Exception("Index creation failed")
        ):
            with pytest.raises(Exception, match="Index creation failed"):
                document_store._create_payload_index(
                    "test", payload_fields_to_index=[{"field_name": "f", "field_schema": {}}]
                )

    @pytest.mark.asyncio
    async def test__create_payload_index_async_handles_exception(self):
        """Test _create_payload_index_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._async_client = MagicMock()

        with patch.object(
            document_store._async_client, "create_payload_index", side_effect=Exception("Index creation failed")
        ):
            with pytest.raises(Exception, match="Index creation failed"):
                await document_store._create_payload_index_async(
                    "test", payload_fields_to_index=[{"field_name": "f", "field_schema": {}}]
                )

    def test__handle_duplicate_documents_handles_exception(self):
        """Test _handle_duplicate_documents exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._client = MagicMock()

        docs = [Document(id="1")]

        document_store._client.retrieve.side_effect = Exception("Retrieve failed")
        with pytest.raises(Exception, match="Retrieve failed"):
            document_store._handle_duplicate_documents(docs, policy=DuplicatePolicy.FAIL)

    @pytest.mark.asyncio
    async def test__handle_duplicate_documents_async_handles_exception(self):
        """Test _handle_duplicate_documents_async exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._async_client = MagicMock()

        docs = [Document(id="1")]

        document_store._async_client.retrieve = AsyncMock(side_effect=Exception("Retrieve failed"))
        with pytest.raises(Exception, match="Retrieve failed"):
            await document_store._handle_duplicate_documents_async(docs, policy=DuplicatePolicy.FAIL)

    def test__drop_duplicate_documents_handles_exception(self):
        """Test _drop_duplicate_documents exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        docs = [Document(id="1"), Document(id="1")]

        result = document_store._drop_duplicate_documents(docs)
        assert len(result) == 1
        assert result[0].id == "1"

    def test__validate_collection_compatibility_handles_exception(self):
        """Test _validate_collection_compatibility exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")
        mock_info = MagicMock()
        mock_info.config.params.vectors = {"some_other_vector": MagicMock()}

        with pytest.raises(QdrantStoreError, match="created outside of Haystack"):
            document_store._validate_collection_compatibility("test", mock_info, rest.Distance.COSINE, 768)

    def test__process_query_point_results_handles_exception(self):
        """Test _process_query_point_results exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")

        results = [MagicMock()]

        with patch(
            "haystack_integrations.document_stores.qdrant.document_store.convert_qdrant_point_to_haystack_document",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(Exception, match="Conversion failed"):
                document_store._process_query_point_results(results)

    def test__process_group_results_handles_exception(self):
        """Test _process_group_results exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")

        group = MagicMock()
        group.hits = [MagicMock()]
        groups = [group]

        with patch(
            "haystack_integrations.document_stores.qdrant.document_store.convert_qdrant_point_to_haystack_document",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(Exception, match="Conversion failed"):
                document_store._process_group_results(groups)

    def test__prepare_collection_config_handles_exception(self):
        """Test _prepare_collection_config exception handling"""
        document_store = QdrantDocumentStore(location=":memory:")

        with pytest.raises(Exception, match="distance"):
            document_store._prepare_collection_config(768, "invalid", False, False, False)


@pytest.mark.integration
class TestQdrantDocumentStore(
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
    GetMetadataFieldsInfoTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=False,
            progress_bar=False,
        )

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test.
        """

        # Check that the lengths of the lists are the same
        assert len(received) == len(expected)

        # Check that the sets are equal, meaning the content and IDs match regardless of order
        assert {doc.id for doc in received} == {doc.id for doc in expected}

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

        # check that the sparse_idf parameter takes effect
        assert hasattr(sparse_config[SPARSE_VECTORS_NAME], "modifier")
        assert sparse_config[SPARSE_VECTORS_NAME].modifier == rest.Modifier.IDF

    def test_query_hybrid(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True, progress_bar=False)

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

        results: list[Document] = document_store._query_hybrid(
            query_sparse_embedding=sparse_embedding, query_embedding=embedding, top_k=10, return_embedding=True
        )
        assert len(results) == 10

        for document in results:
            assert document.sparse_embedding
            assert document.embedding

    def test_query_hybrid_with_group_by(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True, progress_bar=False)

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

        results: list[Document] = document_store._query_hybrid(
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

    def test_update_by_filter_preserves_vectors(self, document_store: QdrantDocumentStore):
        """Test that update_by_filter preserves document embeddings."""
        docs = [
            Document(content="Doc 1", meta={"category": "A"}, embedding=[0.1] * 768),
            Document(content="Doc 2", meta={"category": "B"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        # Update metadata
        updated_count = document_store.update_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 1

        # Verify embedding is preserved
        updated_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(updated_docs) == 1
        assert updated_docs[0].embedding is not None
        assert len(updated_docs[0].embedding) == 768

    def test_get_metadata_field_unique_values_pagination(self, document_store: QdrantDocumentStore):
        """Test getting unique metadata field values with pagination."""
        docs = [Document(content=f"Doc {i}", meta={"value": i % 5}) for i in range(10)]
        document_store.write_documents(docs)

        # Get first 2 unique values
        values_page_1 = document_store.get_metadata_field_unique_values("value", limit=2, offset=0)
        assert len(values_page_1) == 2

        # Get next 2 unique values
        values_page_2 = document_store.get_metadata_field_unique_values("value", limit=2, offset=2)
        assert len(values_page_2) == 2

        # Values should not overlap
        assert set(values_page_1) != set(values_page_2)

    def test_get_metadata_field_unique_values_with_filter(self, document_store: QdrantDocumentStore):
        """Test getting unique metadata field values with filtering."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
        ]
        document_store.write_documents(docs)

        values = document_store.get_metadata_field_unique_values(
            "category", filters={"field": "meta.status", "operator": "==", "value": "active"}
        )
        assert set(values) == {"A", "B"}
