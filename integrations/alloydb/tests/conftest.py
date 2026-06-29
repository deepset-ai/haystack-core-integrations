# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest
from psycopg import Connection

from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore


@pytest.fixture
def document_store(request):
    """
    Integration-test fixture that creates a real AlloyDB connection.

    Requires the following environment variables to be set:
    - ALLOYDB_INSTANCE_URI
    - ALLOYDB_USER
    - ALLOYDB_PASSWORD (unless enable_iam_auth=True)
    - ALLOYDB_DB (optional, defaults to "postgres")
    """
    url = os.environ.get("ALLOYDB_INSTANCE_URI")
    user = os.environ.get("ALLOYDB_USER")
    password = os.environ.get("ALLOYDB_PASSWORD")

    if not all([url, user, password]):
        pytest.skip("Set ALLOYDB_INSTANCE_URI, ALLOYDB_USER and ALLOYDB_PASSWORD to run integration tests")

    table_name = f"haystack_{request.node.name}"
    store = AlloyDBDocumentStore(
        table_name=table_name,
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
        search_strategy="exact_nearest_neighbor",
    )

    yield store

    store._ensure_db_setup()
    store.delete_table()
    store.close()


@pytest.fixture
def document_store_w_hnsw_index(request, monkeypatch):
    monkeypatch.setenv(
        "ALLOYDB_INSTANCE_URI",
        "projects/test-project/locations/us-central1/clusters/test-cluster/instances/test-instance",
    )
    monkeypatch.setenv("ALLOYDB_USER", "postgres")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "postgres")

    table_name = f"haystack_hnsw_{request.node.name}"
    store = AlloyDBDocumentStore(
        table_name=table_name,
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
        search_strategy="hnsw",
    )

    yield store

    store._ensure_db_setup()
    store.delete_table()
    store.close()


@pytest.fixture
def patches_for_unit_tests():
    with (
        patch("haystack_integrations.document_stores.alloydb.document_store.register_vector") as mock_register,
        patch(
            "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore.delete_table"
        ) as mock_delete,
        patch(
            "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore._handle_hnsw"
        ) as mock_hnsw,
        patch("haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore._ensure_db_setup"),
    ):
        yield mock_register, mock_delete, mock_hnsw


@pytest.fixture
def mock_store(patches_for_unit_tests, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv(
        "ALLOYDB_INSTANCE_URI",
        "projects/test-project/locations/us-central1/clusters/test-cluster/instances/test-instance",
    )
    monkeypatch.setenv("ALLOYDB_USER", "postgres")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "postgres")

    store = AlloyDBDocumentStore(
        table_name="haystack",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
        search_strategy="exact_nearest_neighbor",
    )

    yield store


@pytest.fixture
def mock_store_with_mock_connection(mock_store):
    mock_store._connection = Mock(spec=Connection)
    return mock_store
