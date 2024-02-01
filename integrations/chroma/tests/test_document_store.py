# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
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
    LegacyFilterDocumentsTest,
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


class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, LegacyFilterDocumentsTest):
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
        for doc_received, doc_expected in zip(received, expected):
            assert doc_received.content == doc_expected.content
            assert doc_received.meta == doc_expected.meta

    @pytest.mark.unit
    def test_ne_filter(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        """
        We customize this test because Chroma consider "not equal" true when
        a field is missing
        """
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$ne": "100"}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page", "100") != "100"]
        )

    @pytest.mark.unit
    def test_delete_empty(self, document_store: ChromaDocumentStore):
        """
        Deleting a non-existing document should not raise with Chroma
        """
        document_store.delete_documents(["test"])

    @pytest.mark.unit
    def test_delete_not_empty_nonexisting(self, document_store: ChromaDocumentStore):
        """
        Deleting a non-existing document should not raise with Chroma
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        document_store.delete_documents(["non_existing"])

        assert document_store.filter_documents(filters={"id": doc.id}) == [doc]

    @pytest.mark.integration
    def test_to_json(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
        )
        ds_dict = ds.to_dict()
        assert ds_dict == {
            "collection_name": request.node.name,
            "embedding_function": "HuggingFaceEmbeddingFunction",
            "api_key": "1234567890",
        }

    @pytest.mark.integration
    def test_from_json(self):
        collection_name = "test_collection"
        function_name = "HuggingFaceEmbeddingFunction"
        ds_dict = {"collection_name": collection_name, "embedding_function": function_name, "api_key": "1234567890"}

        ds = ChromaDocumentStore.from_dict(ds_dict)
        assert ds._collection_name == collection_name
        assert ds._embedding_function == function_name
        assert ds._embedding_function_params == {"api_key": "1234567890"}

    @pytest.mark.integration
    def test_same_collection_name_reinitialization(self):
        ChromaDocumentStore("test_name")
        ChromaDocumentStore("test_name")

    @pytest.mark.skip(reason="Filter on array contents is not supported.")
    @pytest.mark.unit
    def test_filter_document_array(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on dataframe contents is not supported.")
    @pytest.mark.unit
    def test_filter_document_dataframe(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_eq_filter_table(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_eq_filter_embedding(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$in operator is not supported.")
    @pytest.mark.unit
    def test_in_filter_explicit(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$in operator is not supported. Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_in_filter_table(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$in operator is not supported.")
    @pytest.mark.unit
    def test_in_filter_embedding(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_ne_filter_table(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_ne_filter_embedding(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$nin operator is not supported. Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_nin_filter_table(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$nin operator is not supported. Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_nin_filter_embedding(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$nin operator is not supported.")
    @pytest.mark.unit
    def test_nin_filter(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, document_store: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_multikey_dict(
        self, document_store: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_list(
        self, document_store: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_implicit_and(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_explicit_and(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_implicit_and(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_or(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_or(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_filter_nested_and_or_explicit(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_and_or_implicit(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_or_and(self, document_store: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_multiple_identical_operators_same_level(
        self, document_store: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_duplicate_fail(self, document_store: ChromaDocumentStore):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_duplicate_skip(self, document_store: ChromaDocumentStore):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_duplicate_overwrite(self, document_store: ChromaDocumentStore):
        pass
