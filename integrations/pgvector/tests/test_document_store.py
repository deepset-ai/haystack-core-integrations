# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import psycopg
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack.utils import Secret

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    def test_write_documents(self, document_store: PgvectorDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_blob(self, document_store: PgvectorDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        document_store.write_documents(docs)

        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def test_connection_check_and_recreation(self, document_store: PgvectorDocumentStore):
        document_store._ensure_db_setup()
        original_connection = document_store._connection

        with patch.object(PgvectorDocumentStore, "_connection_is_valid", return_value=False):
            document_store._ensure_db_setup()
            new_connection = document_store._connection

        # verify that a new connection is created
        assert new_connection is not original_connection
        assert document_store._connection == new_connection
        assert original_connection.closed

        assert document_store._cursor is not None
        assert document_store._dict_cursor is not None

        # test with new connection
        with patch.object(PgvectorDocumentStore, "_connection_is_valid", return_value=True):
            document_store._ensure_db_setup()
            same_connection = document_store._connection
            assert same_connection is document_store._connection

    def test_delete_all_documents(self, document_store: PgvectorDocumentStore) -> None:
        document_store.write_documents([Document(id=str(i)) for i in range(10)])
        document_store.delete_all_documents()
        assert document_store.count_documents() == 0
        document_store.write_documents([Document(id="1")])
        assert document_store.count_documents() == 1

    def test_invalid_connection_string(self, monkeypatch):
        monkeypatch.setenv("PG_CONN_STR", "invalid_connection_string")
        document_store = PgvectorDocumentStore()
        with pytest.raises(DocumentStoreError) as e:
            document_store._ensure_db_setup()
        assert "Failed to connect to PostgreSQL database" in str(e)


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_init(monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "some_connection_string")

    document_store = PgvectorDocumentStore(
        create_extension=True,
        schema_name="my_schema",
        table_name="my_table",
        embedding_dimension=512,
        vector_function="l2_distance",
        recreate_table=True,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True,
        hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
        hnsw_index_name="my_hnsw_index",
        hnsw_ef_search=50,
        keyword_index_name="my_keyword_index",
    )

    assert document_store.create_extension
    assert document_store.schema_name == "my_schema"
    assert document_store.table_name == "my_table"
    assert document_store.embedding_dimension == 512
    assert document_store.vector_function == "l2_distance"
    assert document_store.recreate_table
    assert document_store.search_strategy == "hnsw"
    assert document_store.hnsw_recreate_index_if_exists
    assert document_store.hnsw_index_creation_kwargs == {"m": 32, "ef_construction": 128}
    assert document_store.hnsw_index_name == "my_hnsw_index"
    assert document_store.hnsw_ef_search == 50
    assert document_store.keyword_index_name == "my_keyword_index"


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_to_dict(monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "some_connection_string")

    document_store = PgvectorDocumentStore(
        create_extension=False,
        table_name="my_table",
        embedding_dimension=512,
        vector_type="halfvec",
        vector_function="l2_distance",
        recreate_table=True,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True,
        hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
        hnsw_index_name="my_hnsw_index",
        hnsw_ef_search=50,
        keyword_index_name="my_keyword_index",
    )

    assert document_store.to_dict() == {
        "type": "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore",
        "init_parameters": {
            "connection_string": {"env_vars": ["PG_CONN_STR"], "strict": True, "type": "env_var"},
            "create_extension": False,
            "table_name": "my_table",
            "schema_name": "public",
            "embedding_dimension": 512,
            "vector_type": "halfvec",
            "vector_function": "l2_distance",
            "recreate_table": True,
            "search_strategy": "hnsw",
            "hnsw_recreate_index_if_exists": True,
            "language": "english",
            "hnsw_index_creation_kwargs": {"m": 32, "ef_construction": 128},
            "hnsw_index_name": "my_hnsw_index",
            "hnsw_ef_search": 50,
            "keyword_index_name": "my_keyword_index",
        },
    }


@pytest.mark.integration
def test_halfvec_hnsw_write_documents(document_store_w_halfvec_hnsw_index: PgvectorDocumentStore):
    documents = [
        Document(id="1", content="Hello, world!", embedding=[0.1] * 2500),
        Document(id="2", content="Hello, mum!", embedding=[0.3] * 2500),
        Document(id="3", content="Hello, dad!", embedding=[0.2] * 2500),
    ]
    document_store_w_halfvec_hnsw_index.write_documents(documents)

    retrieved_docs = document_store_w_halfvec_hnsw_index.filter_documents()
    retrieved_docs.sort(key=lambda x: x.id)

    for original_doc, retrieved_doc in zip(documents, retrieved_docs, strict=True):
        assert original_doc.id == retrieved_doc.id
        assert original_doc.content == retrieved_doc.content
        assert len(original_doc.embedding) == len(retrieved_doc.embedding)
        # these embeddings are in half precision, so we increase the tolerance
        assert original_doc.embedding == pytest.approx(retrieved_doc.embedding, abs=5e-5)


@pytest.mark.integration
def test_hnsw_index_recreation():
    def get_index_oid(document_store, schema_name, index_name):
        sql_get_index_oid = """
            SELECT c.oid
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'i'
            AND n.nspname = %s
            AND c.relname = %s;
        """
        return document_store._cursor.execute(sql_get_index_oid, (schema_name, index_name)).fetchone()[0]

    # create a new schema
    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    schema_name = "test_schema"
    with psycopg.connect(connection_string, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        conn.execute(f"CREATE SCHEMA {schema_name}")

    # create a first document store and trigger the creation of the hnsw index
    params = {
        "connection_string": Secret.from_token(connection_string),
        "schema_name": schema_name,
        "table_name": "haystack_test_hnsw_index_recreation",
        "search_strategy": "hnsw",
    }
    ds1 = PgvectorDocumentStore(**params)
    ds1._ensure_db_setup()

    # get the hnsw index oid
    hnws_index_name = "haystack_hnsw_index"
    first_oid = get_index_oid(ds1, ds1.schema_name, hnws_index_name)

    # create second document store with recreation enabled
    ds2 = PgvectorDocumentStore(**params, hnsw_recreate_index_if_exists=True)
    ds2._ensure_db_setup()

    # get the index oid
    second_oid = get_index_oid(ds2, ds2.schema_name, hnws_index_name)

    # verify that oids differ
    assert second_oid != first_oid, "Index was not recreated (OID remained the same)"

    # Clean up: drop the schema after the test
    with psycopg.connect(connection_string, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")


@pytest.mark.integration
def test_create_table_if_not_exists():
    def get_table_oid(document_store, schema_name, table_name):
        sql_get_table_oid = """
            SELECT c.oid
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
            AND n.nspname = %s
            AND c.relname = %s;
        """
        return document_store._cursor.execute(sql_get_table_oid, (schema_name, table_name)).fetchone()[0]

    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    schema_name = "test_schema"
    table_name = "test_table"

    # Create a new schema
    with psycopg.connect(connection_string, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        conn.execute(f"CREATE SCHEMA {schema_name}")

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(connection_string),
        schema_name=schema_name,
        table_name=table_name,
    )

    document_store._ensure_db_setup()

    first_table_oid = get_table_oid(document_store, schema_name, table_name)
    assert first_table_oid is not None, "Table was not created"

    document_store._initialize_table()
    second_table_oid = get_table_oid(document_store, schema_name, table_name)
    assert first_table_oid == second_table_oid, "Table was recreated"

    # Clean up: drop the schema after the test
    with psycopg.connect(connection_string, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")


@pytest.mark.integration
def test_delete_table_first_call(document_store):
    """
    Test that delete_table can be executed as the initial operation on the Document Store
    without triggering errors due to an uninitialized state.
    """
    document_store.delete_table()  # if throw error, test fails


@pytest.mark.integration
def test_delete_by_filter(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A", "year": 2023}),
        Document(content="Doc 2", meta={"category": "B", "year": 2023}),
        Document(content="Doc 3", meta={"category": "A", "year": 2024}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 3

    deleted_count = document_store.delete_by_filter(filters={"field": "meta.category", "operator": "==", "value": "A"})
    assert deleted_count == 2
    assert document_store.count_documents() == 1

    remaining_docs = document_store.filter_documents()
    assert len(remaining_docs) == 1
    assert remaining_docs[0].meta["category"] == "B"

    deleted_count = document_store.delete_by_filter(filters={"field": "meta.year", "operator": "==", "value": 2023})
    assert deleted_count == 1
    assert document_store.count_documents() == 0


@pytest.mark.integration
def test_delete_by_filter_no_matches(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A"}),
        Document(content="Doc 2", meta={"category": "B"}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 2

    deleted_count = document_store.delete_by_filter(filters={"field": "meta.category", "operator": "==", "value": "C"})
    assert deleted_count == 0
    assert document_store.count_documents() == 2


@pytest.mark.integration
def test_delete_by_filter_advanced_filters(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A", "year": 2023, "status": "draft"}),
        Document(content="Doc 2", meta={"category": "A", "year": 2024, "status": "published"}),
        Document(content="Doc 3", meta={"category": "B", "year": 2023, "status": "draft"}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 3

    # AND condition
    deleted_count = document_store.delete_by_filter(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "A"},
                {"field": "meta.year", "operator": "==", "value": 2023},
            ],
        }
    )
    assert deleted_count == 1
    assert document_store.count_documents() == 2

    # OR condition
    deleted_count = document_store.delete_by_filter(
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "B"},
                {"field": "meta.status", "operator": "==", "value": "published"},
            ],
        }
    )
    assert deleted_count == 2
    assert document_store.count_documents() == 0


@pytest.mark.integration
def test_update_by_filter(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
        Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
        Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 3

    # update status for category="A" documents
    updated_count = document_store.update_by_filter(
        filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
    )
    assert updated_count == 2

    # verify
    published_docs = document_store.filter_documents(
        filters={"field": "meta.status", "operator": "==", "value": "published"}
    )
    assert len(published_docs) == 2
    for doc in published_docs:
        assert doc.meta["category"] == "A"
        assert doc.meta["status"] == "published"

    # Verify category B still has draft status
    draft_docs = document_store.filter_documents(filters={"field": "meta.status", "operator": "==", "value": "draft"})
    assert len(draft_docs) == 1
    assert draft_docs[0].meta["category"] == "B"


@pytest.mark.integration
def test_update_by_filter_multiple_fields(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A", "year": 2023}),
        Document(content="Doc 2", meta={"category": "A", "year": 2023}),
        Document(content="Doc 3", meta={"category": "B", "year": 2024}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 3

    # update multiple fields for category="A" documents
    updated_count = document_store.update_by_filter(
        filters={"field": "meta.category", "operator": "==", "value": "A"},
        meta={"status": "published", "priority": "high", "reviewed": True},
    )
    assert updated_count == 2

    # verify
    published_docs = document_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "A"})
    assert len(published_docs) == 2
    for doc in published_docs:
        assert doc.meta["status"] == "published"
        assert doc.meta["priority"] == "high"
        assert doc.meta["reviewed"] is True
        assert doc.meta["year"] == 2023  # Original field should still be present

    # verify category B was not updated
    b_docs = document_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "B"})
    assert len(b_docs) == 1
    assert "status" not in b_docs[0].meta
    assert "priority" not in b_docs[0].meta


@pytest.mark.integration
def test_update_by_filter_no_matches(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A"}),
        Document(content="Doc 2", meta={"category": "B"}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 2

    # update documents with category="C" (no matches)
    updated_count = document_store.update_by_filter(
        filters={"field": "meta.category", "operator": "==", "value": "C"}, meta={"status": "published"}
    )
    assert updated_count == 0
    assert document_store.count_documents() == 2

    # verify no documents were updated
    published_docs = document_store.filter_documents(
        filters={"field": "meta.status", "operator": "==", "value": "published"}
    )
    assert len(published_docs) == 0


@pytest.mark.integration
def test_update_by_filter_advanced_filters(document_store: PgvectorDocumentStore):
    docs = [
        Document(content="Doc 1", meta={"category": "A", "year": 2023, "status": "draft"}),
        Document(content="Doc 2", meta={"category": "A", "year": 2024, "status": "draft"}),
        Document(content="Doc 3", meta={"category": "B", "year": 2023, "status": "draft"}),
    ]
    document_store.write_documents(docs)
    assert document_store.count_documents() == 3

    # AND condition
    updated_count = document_store.update_by_filter(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "A"},
                {"field": "meta.year", "operator": "==", "value": 2023},
            ],
        },
        meta={"status": "published"},
    )
    assert updated_count == 1

    # verify only one document was updated
    published_docs = document_store.filter_documents(
        filters={"field": "meta.status", "operator": "==", "value": "published"}
    )
    assert len(published_docs) == 1
    assert published_docs[0].meta["category"] == "A"
    assert published_docs[0].meta["year"] == 2023

    # OR condition
    updated_count = document_store.update_by_filter(
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "B"},
                {"field": "meta.year", "operator": "==", "value": 2024},
            ],
        },
        meta={"featured": True},
    )
    assert updated_count == 2

    featured_docs = document_store.filter_documents(filters={"field": "meta.featured", "operator": "==", "value": True})
    assert len(featured_docs) == 2


@pytest.mark.integration
def test_update_by_filter_empty_meta_raises_error(document_store: PgvectorDocumentStore):
    docs = [Document(content="Doc 1", meta={"category": "A"})]
    document_store.write_documents(docs)

    with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
        document_store.update_by_filter(filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={})
