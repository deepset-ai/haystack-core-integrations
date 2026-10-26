# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest
from psycopg import Connection

from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore


@pytest.fixture
def document_store(request, monkeypatch):
    """
    Integration-test fixture.

    Connects to a real AlloyDB instance when the following environment variables are set:
    - ALLOYDB_INSTANCE_URI
    - ALLOYDB_USER
    - ALLOYDB_PASSWORD

    Otherwise, if `PG_CONN_STR` is set, it connects to a plain PostgreSQL database with the pgvector extension
    (e.g. `postgresql://postgres:postgres@localhost:5432/postgres`) by replacing the AlloyDB Python Connector
    with a direct psycopg connection. AlloyDB is PostgreSQL-compatible, so this tests everything except the
    connector itself.

    Parametrize indirectly with `"document_store_w_hnsw_index"` to use the HNSW search strategy.
    """
    use_alloydb = all(os.environ.get(var) for var in ("ALLOYDB_INSTANCE_URI", "ALLOYDB_USER", "ALLOYDB_PASSWORD"))
    pg_conn_str = os.environ.get("PG_CONN_STR")

    if not use_alloydb:
        if not pg_conn_str:
            pytest.skip(
                "Set ALLOYDB_INSTANCE_URI, ALLOYDB_USER and ALLOYDB_PASSWORD, or PG_CONN_STR, to run integration tests"
            )
        monkeypatch.setattr(AlloyDBDocumentStore, "_create_connection", lambda _: Connection.connect(pg_conn_str))

    use_hnsw = getattr(request, "param", None) == "document_store_w_hnsw_index"
    store = AlloyDBDocumentStore(
        table_name=f"haystack_{request.node.name}",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
        search_strategy="hnsw" if use_hnsw else "exact_nearest_neighbor",
    )

    yield store

    store._ensure_db_setup()
    store.delete_table()
    store.close()


@pytest.fixture
def patches_for_unit_tests():
    with (
        patch(
            "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore.delete_table"
        ) as mock_delete,
        patch(
            "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore._handle_hnsw"
        ) as mock_hnsw,
        patch("haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore._ensure_db_setup"),
    ):
        yield mock_delete, mock_hnsw


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
