# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import operator
import time
import uuid
from unittest import mock

import pytest
from haystack.dataclasses import ByteStream, Document
from haystack.testing.document_store import (
    TEST_EMBEDDING_1,
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
)

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest):
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

        for doc_received, doc_expected in zip(received, expected):
            assert doc_received.content == doc_expected.content
            assert doc_received.meta == doc_expected.meta

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

    def test_invalid_initialization_both_host_and_persist_path(self):
        """
        Test that providing both host and persist_path raises an error.
        """
        with pytest.raises(ValueError):
            store = ChromaDocumentStore(persist_path="./path/to/local/store", host="localhost")
            store._ensure_initialized()

    def test_to_dict(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
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
            },
        }

        ds = ChromaDocumentStore.from_dict(ds_dict)
        assert ds._collection_name == collection_name
        assert ds._embedding_function == function_name
        assert ds._embedding_function_params == {"api_key": "1234567890"}

    def test_same_collection_name_reinitialization(self):
        ChromaDocumentStore("test_1")
        ChromaDocumentStore("test_1")

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

    def test_delete_not_empty_nonexisting(self, document_store: ChromaDocumentStore):
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

    def test_write_documents_unsupported_meta_values(self, document_store: ChromaDocumentStore):
        """
        Unsupported meta values should be removed from the documents before writing them to the database
        """

        docs = [
            Document(content="test doc 1", meta={"invalid": {"dict": "value"}}),
            Document(content="test doc 2", meta={"invalid": ["list", "value"]}),
            Document(content="test doc 3", meta={"ok": 123}),
        ]

        document_store.write_documents(docs)

        written_docs = document_store.filter_documents()
        written_docs.sort(key=lambda x: x.content)

        assert len(written_docs) == 3
        assert [doc.id for doc in written_docs] == [doc.id for doc in docs]
        assert written_docs[0].meta == {}
        assert written_docs[1].meta == {}
        assert written_docs[2].meta == {"ok": 123}

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

    @pytest.mark.integration
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

    def test_delete_all_documents_no_index_recreation(self, document_store: ChromaDocumentStore):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents()
        time.sleep(2)  # need to wait for the deletion to be reflected in count_documents
        assert document_store.count_documents() == 0

        new_doc = Document(id="3", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"
