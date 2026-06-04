# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Integration tests for SupabaseGroongaDocumentStore and SupabaseGroongaBM25Retriever.
#
# These tests require a running stack of three Docker containers defined in
# docker-compose-groonga.yml:
#
#   pgroonga-postgres  PostgreSQL 17 + PGroonga extension (port 5433)
#   postgrest          PostgREST REST API on top of PostgreSQL (port 3000)
#   nginx              Reverse proxy that strips the /rest/v1/ prefix that
#                      supabase-py always appends, then forwards to PostgREST
#                      (port 8000 — the URL used by supabase-py)
#
# Start the stack locally with:
#   make docker-groonga
#
# The test fixture falls back to http://localhost:8000 when SUPABASE_URL is not
# set, so no environment variables are required for local development.
#
# All tests share a single pre-created table (haystack_groonga_test) defined in
# init-pgroonga.sql. PostgREST caches its schema at startup and does not reload
# it for tables created later, so the table must exist before PostgREST starts.
# Data is cleared in fixture teardown instead of recreating the table.

import os

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    DeleteAllTest,
    DeleteByFilterTest,
    DocumentStoreBaseTests,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
)
from haystack.utils import Secret

from haystack_integrations.components.retrievers.supabase import SupabaseGroongaBM25Retriever
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

    def test_count_documents(self, document_store: SupabaseGroongaDocumentStore) -> None:
        docs = [Document(content="doc one"), Document(content="doc two")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

    def test_count_documents_empty(self, document_store: SupabaseGroongaDocumentStore) -> None:
        assert document_store.count_documents() == 0

    def test_write_documents_overwrite(self, document_store: SupabaseGroongaDocumentStore) -> None:
        doc = Document(id="overwrite-id", content="original")
        document_store.write_documents([doc])
        document_store.write_documents(
            [Document(id="overwrite-id", content="updated")], policy=DuplicatePolicy.OVERWRITE
        )
        result = document_store.filter_documents()
        assert len(result) == 1
        assert result[0].content == "updated"

    def test_write_documents_skip_existing(self, document_store: SupabaseGroongaDocumentStore) -> None:
        doc = Document(id="skip-id", content="original")
        document_store.write_documents([doc])
        written = document_store.write_documents(
            [Document(id="skip-id", content="should-be-skipped")], policy=DuplicatePolicy.SKIP
        )
        assert written == 0
        result = document_store.filter_documents()
        assert len(result) == 1
        assert result[0].content == "original"

    def test_write_documents_fail_on_duplicate(self, document_store: SupabaseGroongaDocumentStore) -> None:
        doc = Document(id="fail-id", content="first")
        document_store.write_documents([doc])
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([Document(id="fail-id", content="second")], policy=DuplicatePolicy.FAIL)

    def test_filter_documents(self, document_store: SupabaseGroongaDocumentStore) -> None:
        docs = [Document(content="alpha"), Document(content="beta")]
        document_store.write_documents(docs)
        result = document_store.filter_documents()
        assert len(result) == 2

    def test_filter_documents_with_filters(self, document_store: SupabaseGroongaDocumentStore) -> None:
        docs = [
            Document(content="english doc", meta={"lang": "en"}),
            Document(content="french doc", meta={"lang": "fr"}),
        ]
        document_store.write_documents(docs)
        result = document_store.filter_documents(filters={"field": "meta.lang", "operator": "==", "value": "en"})
        assert len(result) == 1
        assert result[0].meta["lang"] == "en"


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

        retriever = SupabaseGroongaBM25Retriever(document_store=document_store, top_k=5)
        result = retriever.run(query="Python")

        assert "documents" in result
        assert len(result["documents"]) >= 1
        assert any("Python" in doc.content for doc in result["documents"])

    def test_retriever_empty_query(self, document_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=document_store)
        assert retriever.run(query="") == {"documents": []}


@pytest.mark.integration
class TestGroongaDocumentStoreAsyncIntegration:
    """Integration tests for the async methods of SupabaseGroongaDocumentStore."""

    @pytest.fixture
    async def async_store(self, request):  # noqa: ARG002
        supabase_url = os.environ.get("SUPABASE_URL", _LOCAL_SUPABASE_URL)
        service_key = os.environ.get("SUPABASE_SERVICE_KEY", _LOCAL_SERVICE_KEY)
        store = SupabaseGroongaDocumentStore(
            supabase_url=supabase_url,
            supabase_key=Secret.from_token(service_key),
            table_name="haystack_groonga_test",
            recreate_table=False,
        )
        yield store
        await store.delete_all_documents_async()

    async def test_write_and_count_async(self, async_store):
        docs = [Document(content="async doc one"), Document(content="async doc two")]
        written = await async_store.write_documents_async(docs)
        assert written == 2
        count = await async_store.count_documents_async()
        assert count >= 2

    async def test_write_documents_async_empty(self, async_store):
        written = await async_store.write_documents_async([])
        assert written == 0

    async def test_filter_documents_async(self, async_store):
        docs = [
            Document(content="filter async test A", meta={"lang": "en"}),
            Document(content="filter async test B", meta={"lang": "fr"}),
        ]
        await async_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)
        results = await async_store.filter_documents_async()
        assert len(results) >= 2

    async def test_delete_all_documents_async(self, async_store):
        docs = [Document(content="to delete async")]
        await async_store.write_documents_async(docs)
        await async_store.delete_all_documents_async()
        count = await async_store.count_documents_async()
        assert count == 0

    async def test_delete_documents_async_by_id(self, async_store):
        doc = Document(content="delete by id async")
        await async_store.write_documents_async([doc])
        await async_store.delete_documents_async([doc.id])
        results = await async_store.filter_documents_async()
        assert all(d.id != doc.id for d in results)

    async def test_groonga_retrieval_async(self, async_store):
        docs = [
            Document(content="async full-text search with PGroonga"),
            Document(content="synchronous query comparison"),
        ]
        await async_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)
        results = await async_store._groonga_retrieval_async(query="PGroonga", top_k=5)
        assert len(results) >= 1
        assert any("PGroonga" in (d.content or "") for d in results)

    async def test_count_documents_async(self, async_store):
        docs = [Document(content="async one"), Document(content="async two")]
        await async_store.write_documents_async(docs)
        assert await async_store.count_documents_async() == 2

    async def test_count_documents_async_empty(self, async_store):
        assert await async_store.count_documents_async() == 0

    async def test_write_documents_async_overwrite(self, async_store):
        doc = Document(id="async-overwrite-id", content="original")
        await async_store.write_documents_async([doc])
        await async_store.write_documents_async(
            [Document(id="async-overwrite-id", content="updated")], policy=DuplicatePolicy.OVERWRITE
        )
        results = await async_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].content == "updated"

    async def test_write_documents_async_skip_existing(self, async_store):
        doc = Document(id="async-skip-id", content="original")
        await async_store.write_documents_async([doc])
        written = await async_store.write_documents_async(
            [Document(id="async-skip-id", content="should-be-skipped")], policy=DuplicatePolicy.SKIP
        )
        assert written == 0
        results = await async_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].content == "original"

    async def test_write_documents_async_fail_on_duplicate(self, async_store):
        doc = Document(id="async-fail-id", content="first")
        await async_store.write_documents_async([doc])
        with pytest.raises(DuplicateDocumentError):
            await async_store.write_documents_async(
                [Document(id="async-fail-id", content="second")], policy=DuplicatePolicy.FAIL
            )


@pytest.mark.integration
class TestGroongaBM25RetrieverAsyncIntegration:
    """Integration tests for the async run method of SupabaseGroongaBM25Retriever."""

    @pytest.fixture
    async def async_store(self, request):  # noqa: ARG002
        supabase_url = os.environ.get("SUPABASE_URL", _LOCAL_SUPABASE_URL)
        service_key = os.environ.get("SUPABASE_SERVICE_KEY", _LOCAL_SERVICE_KEY)
        store = SupabaseGroongaDocumentStore(
            supabase_url=supabase_url,
            supabase_key=Secret.from_token(service_key),
            table_name="haystack_groonga_test",
            recreate_table=False,
        )
        yield store
        await store.delete_all_documents_async()

    async def test_run_async(self, async_store):
        docs = [
            Document(content="Haystack is a powerful NLP framework"),
            Document(content="Supabase is a Firebase alternative"),
        ]
        await async_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)

        retriever = SupabaseGroongaBM25Retriever(document_store=async_store, top_k=5)
        result = await retriever.run_async(query="Haystack")

        assert "documents" in result
        assert len(result["documents"]) >= 1
        assert any("Haystack" in (d.content or "") for d in result["documents"])

    async def test_run_async_empty_query(self, async_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=async_store)
        result = await retriever.run_async(query="")
        assert result == {"documents": []}

    async def test_run_async_top_k(self, async_store):
        docs = [Document(content=f"async python document {i}") for i in range(4)]
        await async_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)

        retriever = SupabaseGroongaBM25Retriever(document_store=async_store, top_k=2)
        result = await retriever.run_async(query="python")
        assert len(result["documents"]) <= 2
