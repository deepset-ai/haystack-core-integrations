import pytest
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.fixture
def document_store(request):
    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    table_name = f"haystack_{request.node.name}"
    embedding_dimension = 768
    vector_function = "cosine_distance"
    recreate_table = True
    search_strategy = "exact_nearest_neighbor"

    store = PgvectorDocumentStore(
        connection_string=connection_string,
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        vector_function=vector_function,
        recreate_table=recreate_table,
        search_strategy=search_strategy,
    )
    yield store

    store.delete_table()
