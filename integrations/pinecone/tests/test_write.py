import time

import pytest
from haystack import Document
from haystack.document_stores import DuplicatePolicy
from haystack.testing.document_store import (
    WriteDocumentsTest,
)

from pinecone_haystack.document_store import PineconeDocumentStore


class TestWriteDocuments(WriteDocumentsTest):
    def test_write_documents(self, document_store: PineconeDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    # overriden to wait for Pinecone to be updated
    def test_write_documents_duplicate_overwrite(self, document_store: PineconeDocumentStore, sleep_time):
        """
        Test write_documents() overwrites stored Document when trying to write one with same id
        using DuplicatePolicy.OVERWRITE.
        """
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        time.sleep(sleep_time)
        self.assert_documents_are_equal(document_store.filter_documents(), [doc2])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        time.sleep(sleep_time)
        self.assert_documents_are_equal(document_store.filter_documents(), [doc1])

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_fail(self, document_store: PineconeDocumentStore):
        ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_skip(self, document_store: PineconeDocumentStore):
        ...
