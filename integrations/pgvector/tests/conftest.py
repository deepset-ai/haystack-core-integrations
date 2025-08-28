from unittest.mock import patch

import pytest

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.fixture
def document_store(request, monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "postgresql://postgres:postgres@localhost:5432/postgres")
    table_name = f"haystack_{request.node.name}"
    embedding_dimension = 768
    vector_function = "cosine_similarity"
    recreate_table = True
    search_strategy = "exact_nearest_neighbor"

    store = PgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )

    yield store

    # Ensure connection just for table deletion.
    # During test execution, the tested methods are expected to call _ensure_db_setup() themselves.
    store._ensure_db_setup()
    store.delete_table()


@pytest.fixture
def document_store_w_hnsw_index(request, monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "postgresql://postgres:postgres@localhost:5432/postgres")
    table_name = f"haystack_hnsw_{request.node.name}"
    embedding_dimension = 768
    vector_function = "cosine_similarity"
    recreate_table = True
    search_strategy = "hnsw"

    store = PgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )
    yield store

    # Ensure connection just for table deletion.
    # During test execution, the tested methods are expected to call _ensure_db_setup() themselves.
    store._ensure_db_setup()
    store.delete_table()


@pytest.fixture
def document_store_w_halfvec_hnsw_index(request, monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "postgresql://postgres:postgres@localhost:5432/postgres")
    table_name = f"haystack_halfvec_hnsw_{request.node.name}"
    embedding_dimension = 2500
    vector_function = "cosine_similarity"
    recreate_table = True

    store = PgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy="hnsw",
        vector_type="halfvec",
    )
    yield store

    # Ensure connection just for table deletion.
    # During test execution, the tested methods are expected to call _ensure_db_setup() themselves.
    store._ensure_db_setup()
    store.delete_table()


@pytest.fixture
def patches_for_unit_tests():
    with patch("haystack_integrations.document_stores.pgvector.document_store.register_vector") as mock_register, patch(
        "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_table"
    ) as mock_delete, patch(
        "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore._handle_hnsw"
    ) as mock_hnsw:
        yield mock_register, mock_delete, mock_hnsw


@pytest.fixture
def mock_store(patches_for_unit_tests, monkeypatch):  # noqa: ARG001  patches are not explicitly called but necessary
    monkeypatch.setenv("PG_CONN_STR", "some-connection-string")
    table_name = "haystack"
    embedding_dimension = 768
    vector_function = "cosine_similarity"
    recreate_table = True
    search_strategy = "exact_nearest_neighbor"

    store = PgvectorDocumentStore(
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )

    yield store
