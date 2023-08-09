# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import uuid
from typing import List

import numpy as np
import pytest

from chroma_store.document_store import ChromaDocumentStore
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from haystack.testing.preview.document_store import DocumentStoreBaseTests
from haystack.preview import Document


class TestEmbeddingFunction(EmbeddingFunction):
    def __call__(self, _: Documents) -> Embeddings:
        # embed the documents somehow
        return [np.random.default_rng().uniform(-1, 1, 768).tolist()]


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def docstore(self) -> ChromaDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        return ChromaDocumentStore(embedding_function=TestEmbeddingFunction(), collection_name=str(uuid.uuid1()))

    @pytest.mark.unit
    def test_delete_empty(self, docstore: ChromaDocumentStore):
        """
        Deleting a non-existing document should not raise with Chroma
        """
        docstore.delete_documents(["test"])

    @pytest.mark.unit
    def test_delete_not_empty_nonexisting(self, docstore: ChromaDocumentStore):
        """
        This is ok with Chroma
        """
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        docstore.delete_documents(["non_existing"])

        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    @pytest.mark.skip(reason="Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_eq_filter_table(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_eq_filter_embedding(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$in operator is not supported.")
    @pytest.mark.unit
    def test_in_filter_explicit(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$in operator is not supported. Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_in_filter_table(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$in operator is not supported.")
    @pytest.mark.unit
    def test_in_filter_embedding(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_ne_filter_table(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_ne_filter_embedding(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$nin operator is not supported. Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_nin_filter_table(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$nin operator is not supported. Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_nin_filter_embedding(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="$nin operator is not supported.")
    @pytest.mark.unit
    def test_nin_filter(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, docstore: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_multikey_dict(
        self, docstore: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_list(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_implicit_and(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_explicit_and(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_implicit_and(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_simple_or(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_or(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table contents is not supported.")
    @pytest.mark.unit
    def test_filter_nested_and_or_explicit(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_and_or_implicit(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_or_and(self, docstore: ChromaDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter syntax not supported.")
    @pytest.mark.unit
    def test_filter_nested_multiple_identical_operators_same_level(
        self, docstore: ChromaDocumentStore, filterable_docs: List[Document]
    ):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_duplicate_fail(self, docstore: ChromaDocumentStore):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_duplicate_skip(self, docstore: ChromaDocumentStore):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_duplicate_overwrite(self, docstore: ChromaDocumentStore):
        pass
