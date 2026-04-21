# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import operator
import uuid
from unittest import mock

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from haystack.dataclasses import ByteStream, Document, SparseEmbedding
from haystack.testing.document_store import (
    TEST_EMBEDDING_1,
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    FilterDocumentsTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
)

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.document_stores.chroma.errors import ChromaDocumentStoreConfigError
from haystack_integrations.document_stores.chroma.utils import get_embedding_function


@pytest.fixture
def clear_chroma_system_cache():
    """
    Chroma's in-memory client uses a singleton pattern with an internal cache.
    Once a client is created with certain settings, Chroma rejects creating another
    with different settings in the same process. This fixture clears the cache
    before and after tests that use custom client settings.
    """
    SharedSystemClient.clear_system_cache()
    yield
    SharedSystemClient.clear_system_cache()


def test_get_embedding_function_invalid_name_raises():
    with pytest.raises(ChromaDocumentStoreConfigError, match="Invalid function name"):
        get_embedding_function("NonExistentEmbeddingFunction")


class TestDocumentStoreUnit:
    def test_init_in_memory(self):
        store = ChromaDocumentStore()

        assert store._persist_path is None
        assert store._host is None
        assert store._port is None

    def test_init_persistent_storage(self):
        store = ChromaDocumentStore(persist_path="./path/to/local/store")

        assert store._persist_path == "./path/to/local/store"
        assert store._host is None
        assert store._port is None

    def test_init_http_connection(self):
        store = ChromaDocumentStore(host="localhost", port=8000)

        assert store._persist_path is None
        assert store._host == "localhost"
        assert store._port == 8000

    def test_init_with_client_settings(self):
        store = ChromaDocumentStore(client_settings={"anonymized_telemetry": False})
        assert store._client_settings == {"anonymized_telemetry": False}

    def test_invalid_initialization_both_host_and_persist_path(self):
        """
        Test that providing both host and persist_path raises an error.
        """
        with pytest.raises(ValueError):
            store = ChromaDocumentStore(persist_path="./path/to/local/store", host="localhost")
            store._ensure_initialized()

    def test_to_dict(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name,
            embedding_function="HuggingFaceEmbeddingFunction",
            api_key="1234567890",
            client_settings={"anonymized_telemetry": False},
        )
        ds_dict = ds.to_dict()
        assert ds_dict == {
            "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
            "init_parameters": {
                "collection_name": "test_to_dict",
                "embedding_function": "HuggingFaceEmbeddingFunction",
                "persist_path": None,
                "host": None,
                "port": None,
                "api_key": "1234567890",
                "distance_function": "l2",
                "client_settings": {"anonymized_telemetry": False},
            },
        }

    def test_from_dict(self):
        collection_name = "test_collection"
        function_name = "HuggingFaceEmbeddingFunction"
        ds_dict = {
            "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
            "init_parameters": {
                "collection_name": "test_collection",
                "embedding_function": "HuggingFaceEmbeddingFunction",
                "persist_path": None,
                "host": None,
                "port": None,
                "api_key": "1234567890",
                "distance_function": "l2",
                "client_settings": {"anonymized_telemetry": False},
            },
        }

        ds = ChromaDocumentStore.from_dict(ds_dict)
        assert ds._collection_name == collection_name
        assert ds._embedding_function == function_name
        assert ds._embedding_function_params == {"api_key": "1234567890"}
        assert ds._client_settings == {"anonymized_telemetry": False}

    def test_same_collection_name_reinitialization(self):
        ChromaDocumentStore("test_1")
        ChromaDocumentStore("test_1")

    def test_ensure_initialized_invalid_client_settings_raises(self):
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.Settings",
            side_effect=ValueError("bad setting"),
        ):
            store = ChromaDocumentStore(client_settings={"foo": "bar"})
            with pytest.raises(ValueError, match="Invalid client_settings"):
                store._ensure_initialized()

    def test_infer_type_from_value_fallback_for_unknown_type(self):
        assert ChromaDocumentStore._infer_type_from_value(None) == "keyword"
        assert ChromaDocumentStore._infer_type_from_value(["a", "b"]) == "keyword"

    def test_count_unique_metadata_empty_returns_zero_counts(self):
        assert ChromaDocumentStore._count_unique_metadata(None, ["a", "b"]) == {"a": 0, "b": 0}
        assert ChromaDocumentStore._count_unique_metadata([], ["x"]) == {"x": 0}

    def test_compute_field_min_max_skips_non_scalar_values(self):
        metadatas = [{"cat": ["a", "b"]}, {"cat": "X"}, {"cat": "Z"}]
        result = ChromaDocumentStore._compute_field_min_max(metadatas, "cat")
        assert result == {"min": "X", "max": "Z"}

    @pytest.mark.parametrize(
        "result",
        [
            {"ids": ["1"], "documents": None, "metadatas": [{"cat": "A"}]},
            {"ids": ["1"], "documents": ["hello world"], "metadatas": [{"cat": "A"}]},
        ],
        ids=["documents_none", "no_matches"],
    )
    def test_compute_field_unique_values_with_search_term_edge_cases(self, result):
        values, total = ChromaDocumentStore._compute_field_unique_values(result, "cat", "absent", 0, 10)
        assert values == []
        assert total == 0

    def test_filter_metadata_discards_unsupported_types(self, caplog):
        meta = {"ok": "x", "also_ok": None, "bad": {"nested": 1}, "worse": object()}
        with caplog.at_level(logging.WARNING):
            result = ChromaDocumentStore._filter_metadata(meta)
        assert result == {"ok": "x", "also_ok": None}
        assert "bad" in caplog.text and "worse" in caplog.text

    def test_convert_document_to_chroma_rejects_non_document(self):
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            ChromaDocumentStore._convert_document_to_chroma("not a document")  # type: ignore[arg-type]

    def test_convert_document_to_chroma_warns_on_sparse_embedding(self, caplog):
        doc = Document(content="hello", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.1, 0.2]))
        with caplog.at_level(logging.WARNING):
            data = ChromaDocumentStore._convert_document_to_chroma(doc)
        assert data is not None
        assert "sparse_embedding" in caplog.text

    @pytest.mark.parametrize(
        ("result", "expected_embedding"),
        [
            (
                {"ids": ["1"], "documents": ["c"], "metadatas": [{"k": "v"}], "embeddings": [[0.1, 0.2]]},
                [0.1, 0.2],
            ),
            ({"ids": ["1"], "documents": ["c"], "metadatas": [None]}, None),
        ],
        ids=["list_embeddings", "no_embeddings"],
    )
    def test_get_result_to_documents_embedding_variants(self, result, expected_embedding):
        docs = ChromaDocumentStore._get_result_to_documents(result)  # type: ignore[arg-type]
        assert docs[0].embedding == expected_embedding

    @pytest.mark.parametrize(
        ("result", "check"),
        [
            ({"documents": None}, lambda docs: docs == []),
            (
                {"ids": [["a", "b"]], "documents": [["c1", "c2"]], "metadatas": [[{"k": "v"}]]},
                lambda docs: docs[0][0].meta == {"k": "v"} and docs[0][1].meta == {},
            ),
            (
                {"ids": [["a"]], "documents": [["c"]], "metadatas": [[{"k": "v"}]]},
                lambda docs: docs[0][0].embedding is None and docs[0][0].score is None,
            ),
        ],
        ids=["documents_none", "metadata_index_error", "no_embeddings_no_distances"],
    )
    def test_query_result_to_documents_edge_cases(self, result, check):
        assert check(ChromaDocumentStore._query_result_to_documents(result))  # type: ignore[arg-type]


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, embedding_function) -> ChromaDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.get_embedding_function"
        ) as get_func:
            get_func.return_value = embedding_function
            return ChromaDocumentStore(embedding_function="test_function", collection_name=str(uuid.uuid1()))

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method.

        This can happen for example when the Document Store sets a score to returned Documents.
        Since we can't know what the score will be, we can't compare the Documents reliably.
        """
        received.sort(key=operator.attrgetter("id"))
        expected.sort(key=operator.attrgetter("id"))

        for doc_received, doc_expected in zip(received, expected, strict=True):
            assert doc_received.content == doc_expected.content
            assert doc_received.meta == doc_expected.meta

    def test_client_settings_applied(self, clear_chroma_system_cache):
        """
        Chroma's in-memory client uses a singleton pattern with an internal cache.
        Once a client is created with certain settings, Chroma rejects creating another
        with different settings in the same process. We clear the cache before and after
        this test to avoid conflicts with other tests that use default settings.
        """
        store = ChromaDocumentStore(client_settings={"anonymized_telemetry": False})
        store._ensure_initialized()
        assert store._client.get_settings().anonymized_telemetry is False

    def test_distance_metric_initialization(self):
        store = ChromaDocumentStore("test_2", distance_function="cosine")
        store._ensure_initialized()
        assert store._collection.metadata["hnsw:space"] == "cosine"

        with pytest.raises(ValueError):
            ChromaDocumentStore("test_3", distance_function="jaccard")

    def test_distance_metric_reinitialization(self, caplog):
        store = ChromaDocumentStore("test_4", distance_function="cosine")
        store._ensure_initialized()

        with caplog.at_level(logging.WARNING):
            new_store = ChromaDocumentStore("test_4", distance_function="ip")
            new_store._ensure_initialized()

        assert (
            "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
            in caplog.text
        )
        assert store._collection.metadata["hnsw:space"] == "cosine"
        assert new_store._collection.metadata["hnsw:space"] == "cosine"

    def test_metadata_initialization(self, caplog):
        store = ChromaDocumentStore(
            "test_5",
            distance_function="cosine",
            metadata={
                "hnsw:space": "ip",
                "hnsw:search_ef": 101,
                "hnsw:construction_ef": 102,
                "hnsw:M": 103,
            },
        )
        store._ensure_initialized()

        assert store._collection.metadata["hnsw:space"] == "ip"
        assert store._collection.metadata["hnsw:search_ef"] == 101
        assert store._collection.metadata["hnsw:construction_ef"] == 102
        assert store._collection.metadata["hnsw:M"] == 103

        with caplog.at_level(logging.WARNING):
            new_store = ChromaDocumentStore(
                "test_5",
                metadata={
                    "hnsw:space": "l2",
                    "hnsw:search_ef": 101,
                    "hnsw:construction_ef": 102,
                    "hnsw:M": 103,
                },
            )
            new_store._ensure_initialized()

        assert (
            "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
            in caplog.text
        )
        assert store._collection.metadata["hnsw:space"] == "ip"
        assert new_store._collection.metadata["hnsw:space"] == "ip"

    def test_delete_empty(self, document_store: ChromaDocumentStore):
        """
        Deleting a non-existing document should not raise with Chroma
        """
        document_store.delete_documents(["test"])

    def test_delete_not_empty_non_existing(self, document_store: ChromaDocumentStore):
        """
        Deleting a non-existing document should not raise with Chroma
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        document_store.delete_documents(["non_existing"])
        filters = {"operator": "==", "field": "id", "value": doc.id}

        assert document_store.filter_documents(filters=filters)[0].id == doc.id

    def test_filter_documents_return_embeddings(self, document_store: ChromaDocumentStore):
        document_store.write_documents([Document(content="test doc", embedding=TEST_EMBEDDING_1)])

        assert document_store.filter_documents()[0].embedding == pytest.approx(TEST_EMBEDDING_1)

    def test_write_documents_unsupported_meta_values(self, document_store: ChromaDocumentStore) -> None:
        """
        Unsupported meta values should be removed from the documents before writing them to the database
        """

        docs = [
            Document(content="test doc 1", meta={"invalid": {"dict": "value"}}),
            Document(content="test doc 2", meta={"invalid": [["list", "value"]]}),
            Document(content="test doc 3", meta={"invalid": []}),
            Document(content="test doc 4", meta={"invalid": ["list", 2]}),
            Document(content="test doc 5", meta={"ok": 123}),
        ]

        document_store.write_documents(docs)

        written_docs = document_store.filter_documents()
        written_docs.sort(key=lambda x: x.content)

        assert len(written_docs) == 5
        assert [doc.id for doc in written_docs] == [doc.id for doc in docs]
        assert written_docs[0].meta == {}
        assert written_docs[1].meta == {}
        assert written_docs[2].meta == {}
        assert written_docs[3].meta == {}
        assert written_docs[4].meta == {"ok": 123}

    def test_write_documents_supported_meta_values(self, document_store: ChromaDocumentStore) -> None:
        """
        Unsupported meta values should be removed from the documents before writing them to the database
        """

        docs = [
            Document(content="test doc 1", meta={"ok": "test"}),
            Document(content="test doc 2", meta={"ok": ["list", "value"]}),
            Document(content="test doc 3", meta={"ok": [True, False, True]}),
            Document(content="test doc 4", meta={"ok": [1, 2, 3]}),
            Document(content="test doc 5", meta={"ok": [1.1, 2.2, 3.3]}),
            Document(content="test doc 6", meta={"ok": 123}),
        ]

        document_store.write_documents(docs)

        written_docs = document_store.filter_documents()
        written_docs.sort(key=lambda x: x.content)

        assert len(written_docs) == 6
        assert [doc.id for doc in written_docs] == [doc.id for doc in docs]
        assert written_docs[0].meta == {"ok": "test"}
        assert written_docs[1].meta == {"ok": ["list", "value"]}
        assert written_docs[2].meta == {"ok": [True, False, True]}
        assert written_docs[3].meta == {"ok": [1, 2, 3]}
        assert written_docs[4].meta == {"ok": [1.1, 2.2, 3.3]}
        assert written_docs[5].meta == {"ok": 123}

    def test_documents_with_content_none_are_not_stored(self, document_store: ChromaDocumentStore):
        document_store.write_documents([Document(content=None)])
        assert document_store.filter_documents() == []

    def test_blob_not_stored(self, document_store: ChromaDocumentStore):
        bs = ByteStream(data=b"test")
        doc_mixed = Document(content="test", blob=bs)

        document_store.write_documents([doc_mixed])

        retrieved_doc = document_store.filter_documents()[0]
        assert retrieved_doc.content == "test"
        assert retrieved_doc.blob is None

    def test_contains(self, document_store: ChromaDocumentStore, filterable_docs: list[Document]):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "contains", "value": "FOO"}
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.content and "FOO" in doc.content],
        )

    def test_multiple_contains(self, document_store: ChromaDocumentStore, filterable_docs: list[Document]):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "content", "operator": "contains", "value": "FOO"},
                {"field": "content", "operator": "not contains", "value": "BAR"},
            ],
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.content and ("FOO" in doc.content or "BAR" not in doc.content)],
        )

    def test_nested_logical_filters(self, document_store: ChromaDocumentStore, filterable_docs: list[Document]):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.name", "operator": "==", "value": "name_0"},
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "!=", "value": 0},
                        {"field": "meta.page", "operator": "==", "value": "123"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
                        {"field": "meta.date", "operator": "==", "value": "1989-11-09T17:53:00"},
                    ],
                },
            ],
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    # Ensure all required fields are present in doc.meta
                    ("name" in doc.meta and doc.meta.get("name") == "name_0")
                    or (
                        all(key in doc.meta for key in ["number", "page"])
                        and doc.meta.get("number") != 0
                        and doc.meta.get("page") == "123"
                    )
                    or (
                        all(key in doc.meta for key in ["date", "chapter"])
                        and doc.meta.get("chapter") == "conclusion"
                        and doc.meta.get("date") == "1989-11-09T17:53:00"
                    )
                )
            ],
        )

    @pytest.mark.skip(reason="Chroma does not support comparison with null values")
    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with null values")
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with dates")
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with null values")
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with dates")
    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with null values")
    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with dates")
    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with null values")
    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with dates")
    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support comparison with null values")
    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Chroma does not support not operator")
    def test_not_operator(self, document_store, filterable_docs):
        pass

    def test_search(self):
        document_store = ChromaDocumentStore()
        documents = [
            Document(content="First document", meta={"author": "Author1"}),
            Document(content="Second document"),  # No metadata
            Document(content="Third document", meta={"author": "Author2"}),
            Document(content="Fourth document"),  # No metadata
        ]
        document_store.write_documents(documents)
        result = document_store.search(["Third"], top_k=1)

        # Assertions to verify correctness
        assert len(result) == 1
        doc = result[0][0]
        assert doc.content == "Third document"
        assert doc.meta == {"author": "Author2"}
        assert doc.embedding
        assert isinstance(doc.embedding, list)
        assert all(isinstance(el, float) for el in doc.embedding)

        # check that empty filters behave as no filters
        result_empty_filters = document_store.search(["Third"], filters={}, top_k=1)
        assert result == result_empty_filters

    def test_delete_all_documents_index_recreation(self, document_store: ChromaDocumentStore):
        # write some documents
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)

        # get the current document_store config
        config_before = document_store._collection.get(document_store._collection_name)

        # delete all documents with recreating the index
        document_store.delete_all_documents(recreate_index=True)
        assert document_store.count_documents() == 0

        # assure that with the same config
        config_after = document_store._collection.get(document_store._collection_name)

        assert config_before == config_after

        # ensure the collection still exists by writing documents again
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

    def test_search_embeddings(self, document_store: ChromaDocumentStore):
        query_embedding = TEST_EMBEDDING_1
        documents = [
            Document(content="First document", embedding=TEST_EMBEDDING_1, meta={"author": "Author1"}),
            Document(content="Second document", embedding=[0.1] * len(TEST_EMBEDDING_1)),
            Document(content="Third document", embedding=TEST_EMBEDDING_1, meta={"author": "Author2"}),
        ]
        document_store.write_documents(documents)
        result = document_store.search_embeddings([query_embedding], top_k=2)

        # Assertions to verify correctness
        assert len(result) == 1
        assert len(result[0]) == 2
        # The documents with matching embeddings should be returned
        assert all(doc.embedding == pytest.approx(TEST_EMBEDDING_1) for doc in result[0])
        assert all(doc.score is not None for doc in result[0])

        # check that empty filters behave as no filters
        result_empty_filters = document_store.search_embeddings([query_embedding], filters={}, top_k=2)
        assert len(result_empty_filters) == 1
        assert len(result_empty_filters[0]) == 2


@pytest.mark.integration
class TestMetadataOperations:
    """Test new metadata query operations for ChromaDocumentStore"""

    @pytest.fixture
    def document_store(self, embedding_function) -> ChromaDocumentStore:
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.get_embedding_function"
        ) as get_func:
            get_func.return_value = embedding_function
            return ChromaDocumentStore(embedding_function="test_function", collection_name=str(uuid.uuid1()))

    @pytest.fixture
    def populated_store(self, document_store: ChromaDocumentStore) -> ChromaDocumentStore:
        """Fixture with pre-populated test documents with diverse metadata"""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1, "score": 0.9}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2, "score": 0.8}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1, "score": 0.7}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3, "score": 0.95}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2, "score": 0.6}),
            Document(content="Doc 6", meta={"category": "B", "status": "inactive", "priority": 1}),
        ]
        document_store.write_documents(docs)
        return document_store

    def test_get_metadata_fields_info(self, populated_store):
        """Test getting metadata field information"""
        fields_info = populated_store.get_metadata_fields_info()

        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info
        assert "score" in fields_info

        # Check types
        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["status"]["type"] == "keyword"
        assert fields_info["priority"]["type"] == "long"
        assert fields_info["score"]["type"] == "float"

    def test_get_metadata_fields_info_type_inference(self):
        """Test type inference for different data types"""
        store = ChromaDocumentStore()
        docs = [
            Document(
                content="Test", meta={"str_field": "text", "int_field": 42, "float_field": 3.14, "bool_field": True}
            )
        ]
        store.write_documents(docs)

        fields_info = store.get_metadata_fields_info()

        assert fields_info["str_field"]["type"] == "keyword"
        assert fields_info["int_field"]["type"] == "long"
        assert fields_info["float_field"]["type"] == "float"
        assert fields_info["bool_field"]["type"] == "boolean"

    def test_get_metadata_field_min_max_string(self, populated_store):
        """Test getting min/max values for string field (alphabetical)"""
        min_max = populated_store.get_metadata_field_min_max("category")
        assert min_max["min"] == "A"
        assert min_max["max"] == "C"

    def test_get_metadata_field_min_max_missing_field(self, populated_store):
        """Test getting min/max for non-existent field"""
        min_max = populated_store.get_metadata_field_min_max("nonexistent_field")
        assert min_max["min"] is None
        assert min_max["max"] is None

    def test_get_metadata_field_unique_values_basic(self, populated_store):
        """Test getting unique values for metadata field"""
        values, total = populated_store.get_metadata_field_unique_values("category", from_=0, size=10)
        assert sorted(values) == ["A", "B", "C"]
        assert total == 3

    def test_get_metadata_field_unique_values_pagination(self, populated_store):
        """Test pagination of unique values"""
        # First page
        values_page1, total = populated_store.get_metadata_field_unique_values("category", from_=0, size=2)
        assert len(values_page1) == 2
        assert total == 3

        # Second page
        values_page2, total = populated_store.get_metadata_field_unique_values("category", from_=2, size=2)
        assert len(values_page2) == 1
        assert total == 3

        # Check all values are returned across pages
        all_values = values_page1 + values_page2
        assert sorted(all_values) == ["A", "B", "C"]

    def test_get_metadata_field_unique_values_with_search_term(self, populated_store):
        """Test getting unique values filtered by search term"""
        # Search for documents containing "Doc 1"
        values, total = populated_store.get_metadata_field_unique_values(
            "category", search_term="Doc 1", from_=0, size=10
        )
        assert values == ["A"]  # Only Doc 1 has category A
        assert total == 1

    def test_get_metadata_field_unique_values_field_normalization(self, populated_store):
        """Test field name normalization in unique values"""
        # Test with "meta." prefix
        values_with_prefix, total_with_prefix = populated_store.get_metadata_field_unique_values(
            "meta.category", from_=0, size=10
        )
        # Test without "meta." prefix
        values_without_prefix, total_without_prefix = populated_store.get_metadata_field_unique_values(
            "category", from_=0, size=10
        )

        assert sorted(values_with_prefix) == sorted(values_without_prefix) == ["A", "B", "C"]
        assert total_with_prefix == total_without_prefix == 3

    def test_get_metadata_field_unique_values_missing_field(self, populated_store):
        """Test getting unique values for non-existent field"""
        values, total = populated_store.get_metadata_field_unique_values("nonexistent_field", from_=0, size=10)
        assert values == []
        assert total == 0

    def test_get_metadata_field_unique_values_empty_collection(self, document_store):
        """Test getting unique values from empty collection"""
        values, total = document_store.get_metadata_field_unique_values("category", from_=0, size=10)
        assert values == []
        assert total == 0

    def test_get_metadata_field_unique_values_sorting(self, populated_store):
        """Test that unique values are sorted consistently"""
        values, total = populated_store.get_metadata_field_unique_values("status", from_=0, size=10)
        assert values == sorted(values)  # Should be sorted
        assert values == ["active", "inactive"]
        assert total == 2

    def test_get_metadata_field_unique_values_beyond_offset(self, populated_store):
        """Test pagination beyond available results"""
        values, total = populated_store.get_metadata_field_unique_values("category", from_=10, size=10)
        assert values == []  # No values beyond offset
        assert total == 3  # Total count is still 3
