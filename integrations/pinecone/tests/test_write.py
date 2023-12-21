import pytest
from haystack import Document
from haystack.testing.document_store import (
    WriteDocumentsTest,
)

from pinecone_haystack.document_store import PineconeDocumentStore


class TestWriteDocuments(WriteDocumentsTest):
    def test_write_documents(self, document_store: PineconeDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1


    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_fail(self, document_store: PineconeDocumentStore):
        ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_skip(self, document_store: PineconeDocumentStore):
        ...
