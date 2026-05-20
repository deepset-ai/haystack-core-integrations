# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.components.retrievers.supabase import SupabaseGroongaRetriever
from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore


@pytest.fixture()
def document_store():
    """
    Creates a real SupabaseGroongaDocumentStore connected to a test Supabase project.
    Requires SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables to be set.
    """
    store = SupabaseGroongaDocumentStore(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
        table_name="haystack_groonga_integration_test",
        recreate_table=False,
    )
    store.warm_up()
    yield store
    # Cleanup after each test
    all_docs = store.filter_documents()
    if all_docs:
        store.delete_documents([doc.id for doc in all_docs])


@pytest.mark.integration
class TestSupabaseGroongaDocumentStoreIntegration:
    def test_count_empty(self, document_store):
        """Test count is zero for an empty document store."""
        assert document_store.count_documents() == 0

    def test_count_not_empty(self, document_store):
        """Test count is correct after writing documents."""
        docs = [
            Document(content="test doc 1"),
            Document(content="test doc 2"),
            Document(content="test doc 3"),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

    def test_write_and_filter_documents(self, document_store):
        """Test writing and retrieving documents."""
        docs = [
            Document(content="Python is a programming language"),
            Document(content="Haystack is a RAG framework"),
        ]
        written = document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert written == 2

        all_docs = document_store.filter_documents()
        assert len(all_docs) == 2
        contents = {doc.content for doc in all_docs}
        assert "Python is a programming language" in contents
        assert "Haystack is a RAG framework" in contents

    def test_write_documents_duplicate_fail(self, document_store):
        """Test write_documents fails with DuplicatePolicy.FAIL on duplicate."""
        doc = Document(content="test doc")
        document_store.write_documents([doc], policy=DuplicatePolicy.FAIL)

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_write_documents_duplicate_skip(self, document_store):
        """Test write_documents skips duplicate with DuplicatePolicy.SKIP."""
        doc = Document(content="test doc")
        assert document_store.write_documents([doc], policy=DuplicatePolicy.SKIP) == 1
        assert document_store.write_documents([doc], policy=DuplicatePolicy.SKIP) == 0
        assert document_store.count_documents() == 1

    def test_write_documents_duplicate_overwrite(self, document_store):
        """Test write_documents overwrites with DuplicatePolicy.OVERWRITE."""
        doc1 = Document(id="test-id-1", content="original content")
        doc2 = Document(id="test-id-1", content="updated content")

        document_store.write_documents([doc1], policy=DuplicatePolicy.OVERWRITE)
        document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE)

        all_docs = document_store.filter_documents()
        assert len(all_docs) == 1
        assert all_docs[0].content == "updated content"

    def test_delete_documents(self, document_store):
        """Test deleting documents by ID."""
        docs = [
            Document(content="doc to delete"),
            Document(content="doc to keep"),
        ]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert document_store.count_documents() == 2

        document_store.delete_documents([docs[0].id])
        assert document_store.count_documents() == 1

        remaining = document_store.filter_documents()
        assert remaining[0].content == "doc to keep"

    def test_delete_documents_empty_list(self, document_store):
        """Test deleting with empty list does nothing."""
        doc = Document(content="test doc")
        document_store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        document_store.delete_documents([])
        assert document_store.count_documents() == 1

    def test_write_documents_with_meta(self, document_store):
        """Test writing documents with metadata."""
        docs = [
            Document(content="Python tutorial", meta={"language": "en", "topic": "programming"}),
            Document(content="French cooking", meta={"language": "fr", "topic": "cooking"}),
        ]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        all_docs = document_store.filter_documents()
        assert len(all_docs) == 2
        meta_map = {doc.content: doc.meta for doc in all_docs}
        assert meta_map["Python tutorial"]["language"] == "en"
        assert meta_map["French cooking"]["language"] == "fr"

    def test_groonga_retrieval(self, document_store):
        """Test full-text search retrieval using PGroonga."""
        docs = [
            Document(content="Python is a great programming language"),
            Document(content="Haystack is built for RAG pipelines"),
            Document(content="Supabase is a great backend platform"),
        ]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        results = document_store._groonga_retrieval(query="Python", top_k=5)
        assert len(results) >= 1
        assert any("Python" in doc.content for doc in results)

    def test_groonga_retrieval_top_k(self, document_store):
        """Test that top_k limits the number of results."""
        docs = [Document(content=f"document about python number {i}") for i in range(5)]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        results = document_store._groonga_retrieval(query="python", top_k=2)
        assert len(results) <= 2

    def test_retriever_run(self, document_store):
        """Test the retriever returns documents for a query."""
        docs = [
            Document(content="Python programming is fun"),
            Document(content="Java is also popular"),
        ]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        retriever = SupabaseGroongaRetriever(document_store=document_store, top_k=5)
        result = retriever.run(query="Python")

        assert "documents" in result
        assert len(result["documents"]) >= 1
        assert any("Python" in doc.content for doc in result["documents"])

    def test_retriever_empty_query(self, document_store):
        """Test retriever returns empty list for empty query."""
        retriever = SupabaseGroongaRetriever(document_store=document_store)
        result = retriever.run(query="")
        assert result == {"documents": []}