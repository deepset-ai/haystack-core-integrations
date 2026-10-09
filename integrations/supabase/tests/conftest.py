# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haystack_integrations.document_stores.supabase import (
    SupabaseGroongaDocumentStore,
    SupabasePgvectorDocumentStore,
)

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "postgresql://postgres:postgres@localhost:5432/postgres")


@pytest.fixture
def document_store(request, monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", SUPABASE_DB_URL)
    table_name = f"haystack_{request.node.name}"
    embedding_dimension = 768
    vector_function = "cosine_similarity"
    recreate_table = True
    search_strategy = "exact_nearest_neighbor"

    store = SupabasePgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )

    yield store

    store._ensure_db_setup()
    store.delete_table()


@pytest.fixture
def document_store_w_hnsw_index(request, monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", SUPABASE_DB_URL)
    table_name = f"haystack_hnsw_{request.node.name}"
    embedding_dimension = 768
    vector_function = "cosine_similarity"
    recreate_table = True
    search_strategy = "hnsw"

    store = SupabasePgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )
    yield store

    store._ensure_db_setup()
    store.delete_table()


@pytest.fixture
def patches_for_unit_tests():
    with (
        patch("haystack_integrations.document_stores.pgvector.document_store.register_vector") as mock_register,
        patch(
            "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_table"
        ) as mock_delete,
        patch(
            "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore._handle_hnsw"
        ) as mock_hnsw,
    ):
        yield mock_register, mock_delete, mock_hnsw


@pytest.fixture
def mock_store(patches_for_unit_tests, monkeypatch):  # noqa: ARG001  patches are not explicitly called but necessary
    monkeypatch.setenv("SUPABASE_DB_URL", "some-connection-string")
    table_name = "haystack"
    embedding_dimension = 768
    vector_function = "cosine_similarity"
    recreate_table = True
    search_strategy = "exact_nearest_neighbor"

    store = SupabasePgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )

    yield store


GROONGA_STORE_MODULE = "haystack_integrations.document_stores.supabase.groonga_document_store"

# The PostgREST builder methods the Groonga store chains before calling `execute`.
_QUERY_BUILDER_METHODS = ("select", "insert", "upsert", "delete", "eq", "neq", "in_")


@pytest.fixture
def mock_groonga_client():
    """A mocked Supabase client, so the Groonga tests never reach a real database."""
    with patch(f"{GROONGA_STORE_MODULE}.create_client") as mock_create:
        client = MagicMock()
        mock_create.return_value = client

        client.rpc.return_value.execute.return_value = MagicMock(data=[], count=0)

        table = MagicMock()
        client.table.return_value = table
        for method in _QUERY_BUILDER_METHODS:
            getattr(table, method).return_value = table
        table.execute.return_value = MagicMock(data=[], count=0)

        yield client


@pytest.fixture
def mock_async_groonga_client():
    """The async counterpart of `mock_groonga_client`: builder calls stay sync, only `execute` is awaited."""
    with patch(f"{GROONGA_STORE_MODULE}.acreate_client", new_callable=AsyncMock) as mock_acreate:
        client = MagicMock()
        mock_acreate.return_value = client

        client.rpc.return_value.execute = AsyncMock(return_value=MagicMock(data=[], count=0))

        table = MagicMock()
        client.table.return_value = table
        for method in _QUERY_BUILDER_METHODS:
            getattr(table, method).return_value = table
        table.execute = AsyncMock(return_value=MagicMock(data=[], count=0))

        yield client


@pytest.fixture
def mock_groonga_store(mock_groonga_client, monkeypatch) -> SupabaseGroongaDocumentStore:  # noqa: ARG001
    """A warmed-up Groonga store backed by `mock_groonga_client`."""
    store = _groonga_store(monkeypatch)
    store.warm_up()
    return store


@pytest.fixture
def mock_async_groonga_store(mock_async_groonga_client, monkeypatch) -> SupabaseGroongaDocumentStore:  # noqa: ARG001
    """A Groonga store backed by `mock_async_groonga_client`, whose client is created on first async call."""
    return _groonga_store(monkeypatch)


def _groonga_store(monkeypatch) -> SupabaseGroongaDocumentStore:
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    return SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
        table_name="test_groonga_documents",
        recreate_table=False,
    )
