# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import operator
import uuid
from typing import List
from unittest import mock

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from haystack import Document
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
)

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


class _TestEmbeddingFunction(EmbeddingFunction):
    """
    Chroma lets you provide custom functions to compute embeddings,
    we use this feature to provide a fake algorithm returning random
    vectors in unit tests.
    """

    def __call__(self, input: Documents) -> Embeddings:  # noqa - chroma will inspect the signature, it must match
        # embed the documents somehow
        return [np.random.default_rng().uniform(-1, 1, 768).tolist()]


class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self) -> ChromaDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.get_embedding_function"
        ) as get_func:
            get_func.return_value = _TestEmbeddingFunction()
            return ChromaDocumentStore(embedding_function="test_function", collection_name=str(uuid.uuid1()))

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
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
        assert document_store.filter_documents(filters=filters) == [doc]

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

    @pytest.mark.integration
    def test_to_json(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
        )
        ds_dict = ds.to_dict()
        assert ds_dict == {
            "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
            "init_parameters": {
                "collection_name": "test_to_json",
                "embedding_function": "HuggingFaceEmbeddingFunction",
                "persist_path": None,
                "api_key": "1234567890",
                "distance_function": "l2",
            },
        }

    @pytest.mark.integration
    def test_from_json(self):
        collection_name = "test_collection"
        function_name = "HuggingFaceEmbeddingFunction"
        ds_dict = {
            "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
            "init_parameters": {
                "collection_name": "test_collection",
                "embedding_function": "HuggingFaceEmbeddingFunction",
                "persist_path": None,
                "api_key": "1234567890",
                "distance_function": "l2",
            },
        }

        ds = ChromaDocumentStore.from_dict(ds_dict)
        assert ds._collection_name == collection_name
        assert ds._embedding_function == function_name
        assert ds._embedding_function_params == {"api_key": "1234567890"}

    @pytest.mark.integration
    def test_same_collection_name_reinitialization(self):
        ChromaDocumentStore("test_1")
        ChromaDocumentStore("test_1")

    @pytest.mark.integration
    def test_distance_metric_initialization(self):
        store = ChromaDocumentStore("test_2", distance_function="cosine")
        assert store._collection.metadata["hnsw:space"] == "cosine"

        with pytest.raises(ValueError):
            ChromaDocumentStore("test_3", distance_function="jaccard")

    @pytest.mark.integration
    def test_distance_metric_reinitialization(self, caplog):
        store = ChromaDocumentStore("test_4", distance_function="cosine")

        with caplog.at_level(logging.WARNING):
            new_store = ChromaDocumentStore("test_4", distance_function="ip")

        assert (
            "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
            in caplog.text
        )
        assert store._collection.metadata["hnsw:space"] == "cosine"
        assert new_store._collection.metadata["hnsw:space"] == "cosine"

    @pytest.mark.integration
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

        assert (
            "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
            in caplog.text
        )
        assert store._collection.metadata["hnsw:space"] == "ip"
        assert new_store._collection.metadata["hnsw:space"] == "ip"

    def test_contains(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "contains", "value": "FOO"}
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.content and "FOO" in doc.content],
        )

    def test_multiple_contains(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
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

    def test_logical_and_document_filters(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters = [
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                    {"field": "meta.number", "operator": "not in", "value": [2, 9]},
                ],
            },
            {
                "operator": "AND",
                "conditions": [
                    {"field": "content", "operator": "contains", "value": "FOO"},
                    {"field": "content", "operator": "not contains", "value": "BAR"},
                ],
            },
        ]
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "name" in doc.meta
                    and doc.meta.get("name") == "name_0"
                    and "number" in doc.meta
                    and doc.meta.get("number")
                    or [2, 9]
                )
                and (doc.content and ("FOO" in doc.content and "BAR" not in doc.content))
            ],
        )

    # Override inequality tests from FilterDocumentsTest
    # because chroma doesn't return documents with absent meta fields

    def test_comparison_not_equal(self, document_store, filterable_docs):
        """Test filter_documents() with != comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "!=", "value": 100})
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if "number" in d.meta and d.meta.get("number") != 100]
        )

    def test_comparison_not_in(self, document_store, filterable_docs):
        """Test filter_documents() with 'not in' comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "not in", "value": [2, 9]})
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if "number" in d.meta and d.meta.get("number") not in [2, 9]]
        )

    @pytest.mark.skip(reason="Filter on dataframe contents is not supported.")
    def test_comparison_equal_with_dataframe(
        self, document_store: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Filter on dataframe contents is not supported.")
    def test_comparison_not_equal_with_dataframe(
        self, document_store: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

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
