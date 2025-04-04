# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import psycopg
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack.utils import Secret
from packaging.version import parse as parse_version

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


def get_column_type(document_store: PgvectorDocumentStore, schema_name: str, table_name: str, column_name: str) -> str:
    """Helper function to fetch the user-defined type name of a column."""
    sql_get_col_type = """
        SELECT udt_name
        FROM information_schema.columns
        WHERE table_schema = %s
        AND table_name = %s
        AND column_name = %s;
    """
    try:
        result = document_store._cursor.execute(sql_get_col_type, (schema_name, table_name, column_name)).fetchone()
        return result[0] if result else ""
    except Exception:
        # Handle cases where the table might not exist yet or other DB errors during checks
        return ""


def check_pgvector_version(min_version_str="0.7.0"):
    """
    Connects to the test DB and checks if the installed pgvector version
    is less than the required minimum. Returns True if the test should be skipped.
    """
    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    min_version = parse_version(min_version_str)
    try:
        with psycopg.connect(connection_string, autocommit=True, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")
                if cur.fetchone() is None:
                    return True

                cur.execute("SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                if result and result[0]:
                    current_version_str = result[0]
                    current_version = parse_version(current_version_str)
                    should_skip = current_version < min_version
                    if should_skip:
                        pass
                    return should_skip
                else:
                    return True
    except psycopg.Error:
        return True
    except Exception:
        return True


PGVECTOR_VERSION_TOO_LOW = check_pgvector_version("0.7.0")
requires_pgvector_0_7 = pytest.mark.skipif(
    PGVECTOR_VERSION_TOO_LOW, reason="Requires pgvector >= 0.7.0 for halfvec support"
)


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
@requires_pgvector_0_7
def test_halfvec_hnsw_high_dimension():
    """
    Test that PgvectorDocumentStore works with vector_type='halfvec',
    search_strategy='hnsw', and dimensions > 2000.
    """
    high_dimension = 3072
    test_schema_name = "test_halfvec_schema"
    test_table_name = "test_halfvec_table"
    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"

    try:
        with psycopg.connect(connection_string, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {test_schema_name} CASCADE")
            conn.execute(f"CREATE SCHEMA {test_schema_name}")

        document_store = PgvectorDocumentStore(
            connection_string=Secret.from_token(connection_string),
            schema_name=test_schema_name,
            table_name=test_table_name,
            embedding_dimension=high_dimension,
            vector_type="halfvec",
            search_strategy="hnsw",
            vector_function="cosine_similarity",
            recreate_table=True,
            hnsw_recreate_index_if_exists=True,
        )

        # Ensure the database setup is complete
        document_store._ensure_db_setup()

        # Verify the column type
        col_type = get_column_type(document_store, test_schema_name, test_table_name, "embedding")
        assert col_type == "halfvec", f"Expected column type 'halfvec', but found '{col_type}'"

    finally:
        with psycopg.connect(connection_string, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {test_schema_name} CASCADE")
