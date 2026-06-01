# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    DeleteAllTest,
    DeleteByFilterTest,
    DocumentStoreBaseTests,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
)
from haystack.utils import Secret

from haystack_integrations.components.retrievers.supabase import SupabaseGroongaRetriever
from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore


@pytest.fixture()
def document_store(request):
    """
    Creates a real SupabaseGroongaDocumentStore connected to a test Supabase project.
    Requires SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables to be set.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_url:
        pytest.skip("SUPABASE_URL not set")

    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", request.node.name)[:40]
    table_name = f"hg_{safe_name}"

    store = SupabaseGroongaDocumentStore(
        supabase_url=supabase_url,
        supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
        table_name=table_name,
        recreate_table=True,
    )
    store.warm_up()
    yield store
    store.delete_all_documents()


@pytest.mark.integration
class TestSupabaseGroongaDocumentStoreIntegration(
    DocumentStoreBaseTests,
    DeleteAllTest,
    DeleteByFilterTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
):
    pass


@pytest.mark.integration
class TestGroongaRetriever:
    def test_groonga_retrieval(self, document_store):
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
        docs = [Document(content=f"document about python number {i}") for i in range(5)]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        results = document_store._groonga_retrieval(query="python", top_k=2)
        assert len(results) <= 2

    def test_retriever_run(self, document_store):
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
        retriever = SupabaseGroongaRetriever(document_store=document_store)
        assert retriever.run(query="") == {"documents": []}
