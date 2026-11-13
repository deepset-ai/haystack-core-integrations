# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
    @pytest.mark.parametrize("port", [6333, 7333])
    async def test_client_kwargs_are_forwarded_async(self, port):
        document_store = QdrantDocumentStore(
            location=":memory:", port=port, client_kwargs={"check_compatibility": False, "port": 8333}
        )

        with (
            patch(
                "haystack_integrations.document_stores.qdrant.document_store.qdrant_client.AsyncQdrantClient"
            ) as client,
            patch.object(document_store, "_set_up_collection_async", new_callable=AsyncMock),
        ):
            await document_store._initialize_async_client()

        assert client.call_args.kwargs["check_compatibility"] is False
        assert client.call_args.kwargs["port"] == 8333

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

    async def test_multivector_query_wraps_backend_error(self):
        store = QdrantDocumentStore(multivector_embedding_dim=2)
        store._async_client = AsyncMock()
        error = RuntimeError("backend failed")
        store._async_client.query_points.side_effect = error
        with pytest.raises(QdrantStoreError, match="Error querying multivector") as exc:
            await store._query_by_multivector_embedding_async([[1.0, 0.0]])
        assert exc.value.__cause__ is error

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
        mock_collection_info.config.params.sparse_vectors = {SPARSE_VECTORS_NAME: rest.SparseVectorParams()}

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

    async def test_query_rejects_invalid_multivector_embedding(self):
        store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=2)
        with pytest.raises(ValueError, match="multivector_embedding_dim"):
            await store._query_by_multivector_embedding_async([[1.0]])
        assert store._async_client is None

    @pytest.mark.parametrize("top_k", [0, -1])
    async def test_multivector_query_rejects_invalid_top_k(self, top_k):
        store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=2)
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            await store._query_by_multivector_embedding_async([[1.0, 0.0]], top_k=top_k)
        assert store._async_client is None

    async def test_multivector_query_rejects_invalid_filters(self):
        store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=2)
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            await store._query_by_multivector_embedding_async([[1.0, 0.0]], filters={"field": "x"})
        assert store._async_client is None

    async def test_delete_documents_async_invokes_client_and_handles_key_error(self):
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()
        with patch.object(document_store._async_client, "delete") as mock_delete:
            await document_store.delete_documents_async(["doc-1", "doc-2"])
            mock_delete.assert_awaited_once()
        with patch.object(document_store._async_client, "delete", side_effect=KeyError("missing")):
            await document_store.delete_documents_async(["doc-1"])

    @pytest.mark.parametrize(("dimension", "embedding"), [(None, [[1.0, 0.0]]), (2, []), (2, [[1.0]])])
    async def test_write_rejects_invalid_multivector_embedding(self, dimension, embedding):
        store = QdrantDocumentStore(location=":memory:", multivector_embedding_dim=dimension)
        with pytest.raises(ValueError, match="multivector_embedding_dim"):
            await store.write_documents_async([Document(multivector_embedding=embedding)])
        assert await store.count_documents_async() == 0

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("count_documents_async", ()),
            ("delete_all_documents_async", (False,)),
            ("delete_all_documents_async", (True,)),
            ("count_documents_by_filter_async", ({},)),
            ("get_metadata_fields_info_async", ()),
            ("get_metadata_field_min_max_async", ("score",)),
            ("count_unique_metadata_by_filter_async", ({}, ["category"])),
            ("get_metadata_field_unique_values_async", ("category",)),
        ],
    )
    async def test_operations_async_raise_on_client_errors(self, method_name, args):
        document_store = QdrantDocumentStore(location=":memory:")
        await document_store._initialize_async_client()
        err = ValueError("boom")
        with (
            patch.object(document_store._async_client, "count", side_effect=err),
            patch.object(document_store._async_client, "scroll", side_effect=err),
            patch.object(document_store._async_client, "get_collection", side_effect=err),
            patch.object(document_store._async_client, "delete", side_effect=err),
        ):
            with pytest.raises(QdrantStoreError) as exc_info:
                await getattr(document_store, method_name)(*args)
            assert exc_info.value.__cause__ is err

    async def test_get_metadata_fields_info_async_includes_unindexed_fields_when_some_are_indexed(self):
        document_store = QdrantDocumentStore(location=":memory:")
        schema = {"meta.category": rest.PayloadIndexInfo(data_type=rest.PayloadSchemaType.TEXT, points=1)}
        client = AsyncMock()
        client.get_collection.return_value = SimpleNamespace(payload_schema=schema)
        client.scroll.return_value = (
            [SimpleNamespace(payload={"meta": {"category": "news", "priority": 1}})],
            None,
        )
        document_store._async_client = client

        fields = await document_store.get_metadata_fields_info_async()

        assert fields == {"category": {"type": "text"}, "priority": {"type": "long"}}
        client.scroll.assert_awaited_once()

    async def test_close_async(self):
        document_store = QdrantDocumentStore(location=":memory:")
        mock_client = AsyncMock()
        document_store._async_client = mock_client

        await document_store.close_async()

        mock_client.close.assert_awaited_once()
        assert document_store._async_client is None

        await document_store.close_async()
        mock_client.close.assert_awaited_once()

    async def test_close_async_is_exception_safe(self):
        document_store = QdrantDocumentStore(location=":memory:")
        mock_client = AsyncMock()
        mock_client.close.side_effect = RuntimeError("boom")
        document_store._async_client = mock_client

        await document_store.close_async()

        assert document_store._async_client is None

    @pytest.mark.parametrize(
        "method_name, query_kwargs",
        [
            ("_query_by_embedding_async", {"query_embedding": [0.1] * 768}),
            ("_query_by_sparse_async", {"query_sparse_embedding": SparseEmbedding(indices=[0], values=[0.1])}),
        ],
    )
    @pytest.mark.parametrize("scale_score, expected_score", [(True, 0.75), (False, 0.5)])
    async def test_query_with_group_by_scale_score_async(self, method_name, query_kwargs, scale_score, expected_score):
        document_store = QdrantDocumentStore(location=":memory:", similarity="cosine", use_sparse_embeddings=True)
        mock_client = AsyncMock()
        point = rest.ScoredPoint(id=1, version=0, score=0.5, payload={"id": "1", "content": "doc"})
        mock_client.query_points_groups.return_value = rest.GroupsResult(groups=[rest.PointGroup(id="a", hits=[point])])
        document_store._async_client = mock_client

        result = await getattr(document_store, method_name)(
            **query_kwargs, group_by="meta.group", scale_score=scale_score
        )

        assert [doc.score for doc in result] == [expected_score]

    async def test_write_documents_async_progress_bar_counts_documents(self):
        document_store = QdrantDocumentStore(location=":memory:", write_batch_size=2)
        document_store._async_client = AsyncMock()
        with patch("haystack_integrations.document_stores.qdrant.document_store.tqdm") as mock_tqdm:
            await document_store.write_documents_async(
                [Document(content=str(i)) for i in range(3)], policy=DuplicatePolicy.OVERWRITE
            )

        progress_bar = mock_tqdm.return_value.__enter__.return_value
        assert [c.args[0] for c in progress_bar.update.call_args_list] == [2, 1]


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

    async def test_close_async_and_reopen(self, document_store: QdrantDocumentStore):
        assert await document_store.count_documents_async() == 0
        assert document_store._async_client is not None

        await document_store.close_async()

        assert document_store._async_client is None
        assert await document_store.count_documents_async() == 0
        assert document_store._async_client is not None

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

    @pytest.mark.parametrize(
        ("sparse", "dimension", "similarity"),
        [(False, None, "cosine"), (True, None, "dot_product"), (False, 2, "l2"), (True, 2, "cosine")],
    )
    async def test_delete_all_documents_index_recreation_async(self, sparse, dimension, similarity):
        document_store = QdrantDocumentStore(
            location=":memory:",
            use_sparse_embeddings=sparse,
            sparse_idf=sparse,
            multivector_embedding_dim=dimension,
            similarity=similarity,
        )
        await document_store._initialize_async_client()

        docs = [Document(id="1", multivector_embedding=[[1.0, 0.0]] if dimension else None)]
        await document_store.write_documents_async(docs)

        config_before = await document_store._async_client.get_collection(document_store.index)

        await document_store.delete_all_documents_async(recreate_index=True)
        assert await document_store.count_documents_async() == 0

        config_after = await document_store._async_client.get_collection(document_store.index)

        assert config_before.config == config_after.config

        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 1

    async def test_update_by_filter_async_preserves_vectors(self):
        """Test that update_by_filter_async preserves document embeddings."""
        document_store = QdrantDocumentStore(location=":memory:", return_embedding=True, multivector_embedding_dim=2)
        docs = [
            Document(
                content="Doc 1",
                meta={"category": "A"},
                embedding=[0.1] * 768,
                multivector_embedding=[[1.0, 0.0]],
            ),
            Document(content="Doc 2", meta={"category": "B"}, embedding=[0.2] * 768),
        ]
        await document_store.write_documents_async(docs)

        # Update metadata
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 1

        # Verify embedding is preserved
        updated_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(updated_docs) == 1
        assert updated_docs[0].embedding is not None
        assert len(updated_docs[0].embedding) == 768

        assert updated_docs[0].multivector_embedding == [[1.0, 0.0]]

    async def test_count_unique_metadata_by_filter_async_with_meta_prefix(self, document_store: QdrantDocumentStore):
        await document_store.write_documents_async(
            [
                Document(content="1", meta={"category": "A", "status": "draft"}),
                Document(content="2", meta={"category": "B", "status": "draft"}),
                Document(content="3", meta={"category": "A"}),
            ]
        )

        result = await document_store.count_unique_metadata_by_filter_async(
            filters={}, metadata_fields=["meta.category", "status"]
        )

        assert result == {"category": 2, "status": 1}
