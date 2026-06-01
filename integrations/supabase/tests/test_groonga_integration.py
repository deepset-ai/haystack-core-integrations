# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

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

# Defaults for the local Docker stack (docker-compose-groonga.yml).
# PostgREST is configured without a JWT secret, so the key is not validated.
_LOCAL_SUPABASE_URL = "http://localhost:8000"
_LOCAL_SERVICE_KEY = "local-dev-key-not-validated"


def _make_store(request: pytest.FixtureRequest) -> SupabaseGroongaDocumentStore:  # noqa: ARG001
    supabase_url = os.environ.get("SUPABASE_URL", _LOCAL_SUPABASE_URL)
    service_key = os.environ.get("SUPABASE_SERVICE_KEY", _LOCAL_SERVICE_KEY)
    store = SupabaseGroongaDocumentStore(
        supabase_url=supabase_url,
        supabase_key=Secret.from_token(service_key),
        # Fixed table pre-created in init-pgroonga.sql so PostgREST knows about it at startup.
        # Tests clear data in teardown instead of recreating the table.
        table_name="haystack_groonga_test",
        recreate_table=False,
    )
    store.warm_up()
    return store


@pytest.mark.integration
class TestSupabaseGroongaDocumentStoreIntegration(
    DocumentStoreBaseTests,
    DeleteAllTest,
    DeleteByFilterTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
):
    @pytest.fixture
    def document_store(self, request):
        store = _make_store(request)
        yield store
        store.delete_all_documents()

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]) -> None:
        # Embeddings are not stored; strip them and sort by id for order-independent comparison.
        def normalize(doc: Document) -> Document:
            return Document(id=doc.id, content=doc.content, meta=doc.meta)

        assert sorted([normalize(d) for d in received], key=lambda d: d.id or "") == sorted(
            [normalize(d) for d in expected], key=lambda d: d.id or ""
        )

    def test_write_documents(self, document_store: SupabaseGroongaDocumentStore) -> None:
        docs = [
            Document(content="First document", meta={"key": "val"}),
            Document(content="Second document"),
        ]
        assert document_store.write_documents(docs, DuplicatePolicy.FAIL) == len(docs)
        result = document_store.filter_documents()
        self.assert_documents_are_equal(result, docs)


@pytest.mark.integration
class TestGroongaRetriever:
    @pytest.fixture
    def document_store(self, request):
        store = _make_store(request)
        yield store
        store.delete_all_documents()

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
