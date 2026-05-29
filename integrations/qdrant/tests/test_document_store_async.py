from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    _random_embeddings,
)
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)
from qdrant_client.http import models as rest

from haystack_integrations.document_stores.qdrant.document_store import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    QdrantDocumentStore,
    QdrantStoreError,
)


@pytest.mark.asyncio
class TestQdrantDocumentStoreAsyncUnit:
    async def test_query_hybrid_search_batch_failure_async(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        await document_store._initialize_async_client()
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        with patch.object(document_store._async_client, "query_points", side_effect=Exception("query_points")):
            with pytest.raises(QdrantStoreError):
                await document_store._query_hybrid_async(
                    query_sparse_embedding=sparse_embedding, query_embedding=embedding
                )

    async def test_set_up_collection_with_dimension_mismatch_async(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False, similarity="cosine")
        await document_store._initialize_async_client()
        # Mock collection info with different vector size
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = MagicMock()
        mock_collection_info.config.params.vectors.distance = rest.Distance.COSINE
        mock_collection_info.config.params.vectors.size = 512

        with (
            patch.object(document_store._async_client, "collection_exists", return_value=True),
            patch.object(document_store._async_client, "get_collection", return_value=mock_collection_info),
        ):
            with pytest.raises(ValueError, match="different vector size"):
                await document_store._set_up_collection_async("test_collection", 768, False, "cosine", False, False)

    async def test_set_up_collection_with_existing_incompatible_collection_async(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        await document_store._initialize_async_client()
        # Mock collection info with named vectors but missing DENSE_VECTORS_NAME
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = {"some_other_vector": MagicMock()}

        with (
            patch.object(document_store._async_client, "collection_exists", return_value=True),
            patch.object(document_store._async_client, "get_collection", return_value=mock_collection_info),
        ):
            with pytest.raises(QdrantStoreError, match="created outside of Haystack"):
                await document_store._set_up_collection_async("test_collection", 768, False, "cosine", True, False)

    async def test_set_up_collection_use_sparse_embeddings_true_without_named_vectors_async(self):
        """Test that an error is raised when use_sparse_embeddings is True but collection doesn't have named vectors"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        await document_store._initialize_async_client()

        # Mock collection info without named vectors
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = MagicMock(spec=rest.VectorsConfig)

        with (
            patch.object(document_store._async_client, "collection_exists", return_value=True),
            patch.object(document_store._async_client, "get_collection", return_value=mock_collection_info),
        ):
            with pytest.raises(QdrantStoreError, match="without sparse embedding vectors"):
                await document_store._set_up_collection_async("test_collection", 768, False, "cosine", True, False)

    async def test_set_up_collection_use_sparse_embeddings_false_with_named_vectors_async(self):
        """Test that an error is raised when use_sparse_embeddings is False but collection has named vectors"""
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False)
        await document_store._initialize_async_client()
        # Mock collection info with named vectors
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = {DENSE_VECTORS_NAME: MagicMock()}

        with (
            patch.object(document_store._async_client, "collection_exists", return_value=True),
            patch.object(document_store._async_client, "get_collection", return_value=mock_collection_info),
        ):
            with pytest.raises(QdrantStoreError, match="with sparse embedding vectors"):
                await document_store._set_up_collection_async("test_collection", 768, False, "cosine", False, False)

    async def test_set_up_collection_with_distance_mismatch_async(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False, similarity="cosine")
        await document_store._initialize_async_client()

        # Mock collection info with different distance
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = MagicMock()
        mock_collection_info.config.params.vectors.distance = rest.Distance.DOT
        mock_collection_info.config.params.vectors.size = 768

        with (
            patch.object(document_store._async_client, "collection_exists", return_value=True),
            patch.object(document_store._async_client, "get_collection", return_value=mock_collection_info),
        ):
            with pytest.raises(ValueError, match="different similarity"):
                await document_store._set_up_collection_async("test_collection", 768, False, "cosine", False, False)

    async def test_query_by_sparse_async_raises_when_sparse_disabled(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False)
        sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.1, 0.2])
        with pytest.raises(QdrantStoreError, match="use_sparse_embeddings=False"):
            await document_store._query_by_sparse_async(query_sparse_embedding=sparse_embedding)

    async def test_delete_documents_async_invokes_client_and_handles_key_error(self):
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()
        with patch.object(document_store._async_client, "delete") as mock_delete:
            await document_store.delete_documents_async(["doc-1", "doc-2"])
            mock_delete.assert_awaited_once()
        with patch.object(document_store._async_client, "delete", side_effect=KeyError("missing")):
            await document_store.delete_documents_async(["doc-1"])

    @pytest.mark.parametrize(
        ("method_name", "args", "expected"),
        [
            ("count_documents_async", (), 0),
            ("count_documents_by_filter_async", ({},), 0),
            ("get_metadata_fields_info_async", (), {}),
            ("get_metadata_field_min_max_async", ("score",), {}),
            ("count_unique_metadata_by_filter_async", ({}, ["category"]), {"category": 0}),
            ("get_metadata_field_unique_values_async", ("category",), []),
        ],
    )
    async def test_metadata_methods_async_absorb_client_errors(self, method_name, args, expected):
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()
        err = ValueError("boom")
        with (
            patch.object(document_store._async_client, "count", side_effect=err),
            patch.object(document_store._async_client, "scroll", side_effect=err),
            patch.object(document_store._async_client, "get_collection", side_effect=err),
        ):
            assert await getattr(document_store, method_name)(*args) == expected


@pytest.mark.integration
@pytest.mark.asyncio
class TestQdrantDocumentStoreAsync(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @pytest_asyncio.fixture
    async def document_store(self):
        store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=False,
            progress_bar=False,
        )
        yield store

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        assert len(received) == len(expected)
        assert {doc.id for doc in received} == {doc.id for doc in expected}

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store: QdrantDocumentStore):
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    async def test_sparse_configuration_async(self):
        document_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            use_sparse_embeddings=True,
            sparse_idf=True,
        )
        await document_store._initialize_async_client()

        collection = await document_store._async_client.get_collection("Document")
        sparse_config = collection.config.params.sparse_vectors

        assert SPARSE_VECTORS_NAME in sparse_config
        # check that the `sparse_idf` parameter takes effect
        assert hasattr(sparse_config[SPARSE_VECTORS_NAME], "modifier")
        assert sparse_config[SPARSE_VECTORS_NAME].modifier == rest.Modifier.IDF

    async def test_query_hybrid_async(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True, progress_bar=False)

        docs = []
        for i in range(20):
            docs.append(
                Document(
                    content=f"doc {i}", sparse_embedding=generate_sparse_embedding(), embedding=_random_embeddings(768)
                )
            )

        await document_store.write_documents_async(docs)
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        results: list[Document] = await document_store._query_hybrid_async(
            query_sparse_embedding=sparse_embedding, query_embedding=embedding, top_k=10, return_embedding=True
        )
        assert len(results) == 10

        for document in results:
            assert document.sparse_embedding
            assert document.embedding

    async def test_query_hybrid_with_group_by_async(self, generate_sparse_embedding):
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

        await document_store.write_documents_async(docs)

        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        results: list[Document] = await document_store._query_hybrid_async(
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

    async def test_query_hybrid_fail_without_sparse_embedding_async(self, document_store):
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        with pytest.raises(QdrantStoreError):
            await document_store._query_hybrid_async(
                query_sparse_embedding=sparse_embedding,
                query_embedding=embedding,
            )

    async def test_update_by_filter_async_preserves_vectors(self, document_store: QdrantDocumentStore):
        """Test that update_by_filter_async preserves document embeddings."""
        docs = [
            Document(content="Doc 1", meta={"category": "A"}, embedding=[0.1] * 768),
            Document(content="Doc 2", meta={"category": "B"}, embedding=[0.2] * 768),
        ]
        await document_store.write_documents_async(docs)

        # Update metadata
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 1

        # Verify embedding is preserved
        updated_docs = []
        async for doc in document_store._get_documents_generator_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        ):
            updated_docs.append(doc)
        assert len(updated_docs) == 1
        assert updated_docs[0].embedding is not None
        assert len(updated_docs[0].embedding) == 768
