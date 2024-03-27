# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest

from src.haystack_integrations.document_stores.supabase import SupabaseDocumentStore


class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest):

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        for doc in received:
            # Supabase seems to convert integers to floats
            # We convert them back to integers to compare them
            if "number" in doc.meta:
                doc.meta["number"] = int(doc.meta["number"])

        # Lists comparison
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected):
            assert received_doc.meta == expected_doc.meta
            assert received_doc.content == expected_doc.content
            if received_doc.dataframe is None:
                assert expected_doc.dataframe is None
            else:
                assert received_doc.dataframe.equals(expected_doc.dataframe)
            # unfortunately, Supabase returns a slightly different embedding
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)

    def test_count_empty(self, document_store: DocumentStore):
        return super().test_count_empty(document_store)

    def test_count_not_empty(self, document_store: DocumentStore):
        assert document_store.count_documents() == 0
        document_store.write_documents(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert document_store.count_documents() == 3

    def test_delete_documents(self, document_store: DocumentStore):
        return super().test_delete_documents(document_store)

    def test_delete_documents_empty_document_store(self, document_store: DocumentStore):
        return super().test_delete_documents_empty_document_store(document_store)

    def test_delete_documents_non_existing_document(self, document_store: DocumentStore):
        return super().test_delete_documents_non_existing_document(document_store)

    def test_embedding_retrieval(self, document_store: SupabaseDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = np.random.rand(768).tolist()

        assert document_store.count_documents() == 0
        docs = [
            Document(content="Most similar document", embedding=most_similar_embedding),
            Document(content="2nd best document", embedding=second_best_embedding),
            Document(content="Not very similar document", embedding=another_embedding),
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 3
        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    @patch("src.haystack_integrations.document_stores.supabase.document_store.vecs")
    def test_init(self, mock_supabase):

        document_store = SupabaseDocumentStore(host="fake-host", password="password", dimension=30)

        user: str = "postgres"
        port: str = "5432"
        db_name: str = "postgres"

        db_connection = f"postgresql://{user}:password@fake-host:{port}/{db_name}"
        mock_supabase.create_client.assert_called_with(db_connection)

        assert document_store._collection_name == "documents"
        assert document_store.dimension == 30

    @pytest.mark.skip(reason="Supabase only supports UPSERT operations")
    def test_write_documents_duplicate_fail(self, document_store: SupabaseDocumentStore): ...

    @pytest.mark.skip(reason="Supabase only supports UPSERT operations")
    def test_write_documents_duplicate_skip(self, document_store: SupabaseDocumentStore): ...

    def test_write_documents_duplicate_overwrite(self, document_store: SupabaseDocumentStore):
        """
        Test write_documents() overwrites stored Document when trying to write one with same id
        using DuplicatePolicy.OVERWRITE.
        """
        embedding = [0.0] * 768
        doc1 = Document(id="1", content="test doc 1", embedding=[0.1] * 768)
        doc2 = Document(id="1", content="test doc 2", embedding=embedding)

        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc2])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc1])

    def test_write_documents(self, document_store: DocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    def test_write_documents_invalid_input(self, document_store: DocumentStore):
        return super().test_write_documents_invalid_input(document_store)
