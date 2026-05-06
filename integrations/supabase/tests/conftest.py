# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore

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
