from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

from haystack_integrations.document_stores.qdrant.document_store import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    QdrantDocumentStore,
    QdrantStoreError,
    get_batches_from_generator,
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

    def test_get_distance_known(self):
        document_store = QdrantDocumentStore(location=":memory:")
        assert document_store.get_distance("cosine") == rest.Distance.COSINE
        assert document_store.get_distance("dot_product") == rest.Distance.DOT
        assert document_store.get_distance("l2") == rest.Distance.EUCLID

    def test_get_distance_unknown_raises(self):
        document_store = QdrantDocumentStore(location=":memory:")
        with pytest.raises(QdrantStoreError, match="not supported"):
            document_store.get_distance("unknown")

    def test_validate_filters_accepts_dict_and_native(self):
        QdrantDocumentStore._validate_filters(None)
        QdrantDocumentStore._validate_filters({"operator": "==", "field": "meta.x", "value": 1})
        QdrantDocumentStore._validate_filters(rest.Filter(must=[]))

    def test_validate_filters_rejects_non_dict_non_filter(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            QdrantDocumentStore._validate_filters("not-a-filter")

    def test_validate_filters_rejects_dict_without_operator(self):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            QdrantDocumentStore._validate_filters({"field": "meta.x"})

    def test_check_stop_scrolling(self):
        assert QdrantDocumentStore._check_stop_scrolling(None) is True
        empty_offset = SimpleNamespace(num=0, uuid="")
        assert QdrantDocumentStore._check_stop_scrolling(empty_offset) is True
        non_empty_offset = SimpleNamespace(num=5, uuid="abc")
        assert QdrantDocumentStore._check_stop_scrolling(non_empty_offset) is False

    def test_infer_type_from_value(self):
        assert QdrantDocumentStore._infer_type_from_value(True) == "boolean"
        assert QdrantDocumentStore._infer_type_from_value(1) == "long"
        assert QdrantDocumentStore._infer_type_from_value(1.5) == "float"
        assert QdrantDocumentStore._infer_type_from_value("x") == "keyword"
        assert QdrantDocumentStore._infer_type_from_value([1, 2]) == "keyword"

    def test_process_records_fields_info(self):
        records = [
            SimpleNamespace(payload={"meta": {"category": "A", "score": 0.9, "missing": None}}),
            SimpleNamespace(payload={"meta": {"category": "B"}}),  # category already seen
            SimpleNamespace(payload=None),  # no payload
            SimpleNamespace(payload={"other": "noise"}),  # no meta
        ]
        field_info: dict = {}
        QdrantDocumentStore._process_records_fields_info(records, field_info)
        assert field_info == {"category": {"type": "keyword"}, "score": {"type": "float"}}

    def test_metadata_fields_info_from_schema(self):
        schema = {
            "meta.category": SimpleNamespace(data_type="keyword"),
            "meta.priority": SimpleNamespace(data_type="integer"),
            "meta.unknown": object(),  # no data_type attribute
            "not_meta_prefixed": SimpleNamespace(data_type="keyword"),
        }
        fields = QdrantDocumentStore._metadata_fields_info_from_schema(schema)
        assert fields == {
            "category": {"type": "keyword"},
            "priority": {"type": "integer"},
            "unknown": {"type": "unknown"},
        }

    def test_process_records_min_max(self):
        records = [
            SimpleNamespace(payload={"meta": {"score": 0.5}}),
            SimpleNamespace(payload={"meta": {"score": 0.9}}),
            SimpleNamespace(payload={"meta": {"score": None}}),
            SimpleNamespace(payload={"meta": {"other": 100}}),
            SimpleNamespace(payload=None),
        ]
        min_v, max_v = QdrantDocumentStore._process_records_min_max(records, "score", None, None)
        assert min_v == 0.5
        assert max_v == 0.9

    def test_process_records_count_unique(self):
        records = [
            SimpleNamespace(payload={"meta": {"category": "A", "tags": ["x"]}}),
            SimpleNamespace(payload={"meta": {"category": "B", "tags": ["x"]}}),
            SimpleNamespace(payload={"meta": {"category": "A", "tags": ["y"]}}),
            SimpleNamespace(payload=None),
        ]
        unique: dict = {"category": set(), "tags": set()}
        QdrantDocumentStore._process_records_count_unique(records, ["category", "tags"], unique)
        assert unique["category"] == {"A", "B"}
        assert unique["tags"] == {"['x']", "['y']"}

    def test_process_records_unique_values_stops_when_filled(self):
        records = [SimpleNamespace(payload={"meta": {"v": i}}) for i in range(10)]
        values: list = []
        values_set: set = set()
        done = QdrantDocumentStore._process_records_unique_values(records, "v", values, values_set, offset=0, limit=3)
        assert done is True
        assert values[:3] == [0, 1, 2]

    def test_process_records_unique_values_not_done(self):
        records = [SimpleNamespace(payload={"meta": {"v": 1}}), SimpleNamespace(payload=None)]
        values: list = []
        values_set: set = set()
        done = QdrantDocumentStore._process_records_unique_values(records, "v", values, values_set, offset=0, limit=5)
        assert done is False
        assert values == [1]

    def test_create_updated_point_from_record_adds_missing_meta(self):
        record = SimpleNamespace(
            id="abc",
            payload={"content": "hello"},
            vector=[0.1, 0.2],
        )
        point = QdrantDocumentStore._create_updated_point_from_record(record, {"status": "published"})
        assert point.payload["meta"] == {"status": "published"}
        assert point.payload["content"] == "hello"
        assert point.vector == [0.1, 0.2]

    def test_create_updated_point_from_record_merges_meta(self):
        record = SimpleNamespace(
            id="abc",
            payload={"content": "hello", "meta": {"category": "A"}},
            vector=None,
        )
        point = QdrantDocumentStore._create_updated_point_from_record(record, {"status": "published"})
        assert point.payload["meta"] == {"category": "A", "status": "published"}
        assert point.vector == {}

    def test_drop_duplicate_documents(self):
        document_store = QdrantDocumentStore(location=":memory:")
        doc1 = Document(id="1", content="a")
        doc2 = Document(id="2", content="b")
        doc1_dup = Document(id="1", content="a")
        result = document_store._drop_duplicate_documents([doc1, doc2, doc1_dup])
        assert [d.id for d in result] == ["1", "2"]

    def test_prepare_collection_config_without_sparse(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False)
        vectors_config, sparse_config = document_store._prepare_collection_config(
            embedding_dim=768, distance=rest.Distance.COSINE
        )
        assert isinstance(vectors_config, rest.VectorParams)
        assert sparse_config is None

    def test_prepare_collection_config_with_sparse_and_idf(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
        vectors_config, sparse_config = document_store._prepare_collection_config(
            embedding_dim=768, distance=rest.Distance.COSINE, sparse_idf=True
        )
        assert DENSE_VECTORS_NAME in vectors_config
        assert sparse_config[SPARSE_VECTORS_NAME].modifier == rest.Modifier.IDF

    def test_prepare_client_params_does_not_mutate_metadata(self):
        metadata = {"key": "value"}
        document_store = QdrantDocumentStore(location=":memory:", metadata=metadata)
        params = document_store._prepare_client_params()
        params["metadata"]["added"] = "x"
        assert metadata == {"key": "value"}

    def test_get_batches_from_generator(self):
        batches = list(get_batches_from_generator([1, 2, 3, 4, 5], 2))
        assert batches == [(1, 2), (3, 4), (5,)]
        assert list(get_batches_from_generator([], 2)) == []

    def test_query_by_sparse_raises_when_sparse_disabled(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=False)
        sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.1, 0.2])
        with pytest.raises(QdrantStoreError, match="use_sparse_embeddings=False"):
            document_store._query_by_sparse(query_sparse_embedding=sparse_embedding)

    @pytest.mark.parametrize(
        ("method_name", "args", "expected"),
        [
            ("count_documents", (), 0),
            ("count_documents_by_filter", ({},), 0),
            ("get_metadata_fields_info", (), {}),
            ("get_metadata_field_min_max", ("score",), {}),
            ("count_unique_metadata_by_filter", ({}, ["category"]), {"category": 0}),
            ("get_metadata_field_unique_values", ("category",), []),
        ],
    )
    def test_metadata_methods_swallow_client_errors(self, method_name, args, expected):
        document_store = QdrantDocumentStore(location=":memory:")
        document_store._initialize_client()
        err = ValueError("boom")
        with (
            patch.object(document_store._client, "count", side_effect=err),
            patch.object(document_store._client, "scroll", side_effect=err),
            patch.object(document_store._client, "get_collection", side_effect=err),
        ):
            assert getattr(document_store, method_name)(*args) == expected


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

        # check that the `sparse_idf` parameter takes effect
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
