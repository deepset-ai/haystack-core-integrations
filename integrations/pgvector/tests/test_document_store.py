# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import psycopg
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
)
from haystack.utils import Secret

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    WriteDocumentsTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    def test_get_metadata_fields_info_empty_collection(self, document_store: PgvectorDocumentStore):
        """PgvectorDocumentStore always includes 'content' in fields info, even for empty stores."""
        assert document_store.count_documents() == 0

        fields_info = document_store.get_metadata_fields_info()
        assert fields_info == {"content": {"type": "text"}}

    def test_get_metadata_field_min_max_empty_collection(self, document_store: PgvectorDocumentStore):
        """PgvectorDocumentStore raises ValueError when the field doesn't exist in the store."""
        assert document_store.count_documents() == 0

        with pytest.raises(ValueError, match="not found in document store"):
            document_store.get_metadata_field_min_max("priority")

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
def test_update_by_filter_empty_meta_raises_error(document_store: PgvectorDocumentStore):
    docs = [Document(content="Doc 1", meta={"category": "A"})]
    document_store.write_documents(docs)

    with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
        document_store.update_by_filter(filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={})


@pytest.mark.integration
def test_get_metadata_field_min_max(document_store: PgvectorDocumentStore):
    # Test with text/string values - lexicographic comparison
    text_docs = [
        Document(content="Doc 1", meta={"category": "Zebra", "status": "active"}),
        Document(content="Doc 2", meta={"category": "Apple", "status": "pending"}),
        Document(content="Doc 3", meta={"category": "Banana", "status": "inactive"}),
        Document(content="Doc 4", meta={"category": "apple", "status": "active"}),
    ]
    document_store.write_documents(text_docs)

    # Test lexicographic min/max for text fields (case-sensitive)
    min_max_category = document_store.get_metadata_field_min_max("meta.category")
    assert min_max_category["min"] == "Apple"  # 'A' comes before 'B' and 'Z' and 'a'
    assert min_max_category["max"] == "apple"  # 'a' comes after 'A', 'B', 'Z' in ASCII

    min_max_status = document_store.get_metadata_field_min_max("meta.status")
    assert min_max_status["min"] == "active"  # 'a' comes before 'i' and 'p'
    assert min_max_status["max"] == "pending"  # 'p' comes after 'a' and 'i'

    # Test with empty strings
    empty_string_docs = [
        Document(content="Doc 1", meta={"tag": ""}),
        Document(content="Doc 2", meta={"tag": "A"}),
        Document(content="Doc 3", meta={"tag": "B"}),
    ]
    document_store.write_documents(empty_string_docs)
    min_max_tag = document_store.get_metadata_field_min_max("meta.tag")
    assert min_max_tag["min"] == ""  # Empty string is typically minimum
    assert min_max_tag["max"] == "B"  # 'B' is maximum

    # Test with special characters
    special_char_docs = [
        Document(content="Doc 1", meta={"code": "!@#"}),
        Document(content="Doc 2", meta={"code": "$%^"}),
        Document(content="Doc 3", meta={"code": "&*()"}),
    ]
    document_store.write_documents(special_char_docs)
    min_max_code = document_store.get_metadata_field_min_max("meta.code")
    # Special characters have specific ASCII ordering
    assert min_max_code["min"] in ["!@#", "$%^", "&*()"]
    assert min_max_code["max"] in ["!@#", "$%^", "&*()"]

    # Test with Unicode characters
    unicode_docs = [
        Document(content="Doc 1", meta={"name": "Ángel"}),
        Document(content="Doc 2", meta={"name": "Zebra"}),
        Document(content="Doc 3", meta={"name": "Alpha"}),
    ]
    document_store.write_documents(unicode_docs)
    min_max_name = document_store.get_metadata_field_min_max("meta.name")
    # With COLLATE "C", comparison is byte-order based
    # "Alpha" should be minimum (A comes first in ASCII)
    # "Ángel" or "Zebra" will be maximum depending on byte encoding
    assert min_max_name["min"] == "Alpha"  # 'A' comes first in ASCII
    assert min_max_name["max"] in ["Ángel", "Zebra"]  # Depends on UTF-8 byte encoding


@pytest.mark.integration
def test_get_metadata_field_unique_values(document_store: PgvectorDocumentStore):
    # Test with string values
    docs = [
        Document(content="Python programming", meta={"category": "A", "language": "Python"}),
        Document(content="Java programming", meta={"category": "B", "language": "Java"}),
        Document(content="Python scripting", meta={"category": "A", "language": "Python"}),
        Document(content="JavaScript development", meta={"category": "C", "language": "JavaScript"}),
        Document(content="Python data science", meta={"category": "A", "language": "Python"}),
        Document(content="Java backend", meta={"category": "B", "language": "Java"}),
    ]
    document_store.write_documents(docs)

    # Test getting all unique values without search term
    unique_values, total_count = document_store.get_metadata_field_unique_values("meta.category", None, 0, 10)
    assert set(unique_values) == {"A", "B", "C"}
    assert total_count == 3

    # Test with "meta." prefix
    unique_languages, total_languages = document_store.get_metadata_field_unique_values("meta.language", None, 0, 10)
    assert set(unique_languages) == {"Python", "Java", "JavaScript"}
    assert total_languages == 3

    # Test pagination - first page
    unique_values_page1, total_count_page1 = document_store.get_metadata_field_unique_values(
        "meta.category", None, 0, 2
    )
    assert len(unique_values_page1) == 2
    assert all(val in ["A", "B", "C"] for val in unique_values_page1)
    assert total_count_page1 == 3

    # Test pagination - second page
    unique_values_page2, total_count_page2 = document_store.get_metadata_field_unique_values(
        "meta.category", None, 2, 2
    )
    assert len(unique_values_page2) == 1
    assert unique_values_page2[0] in ["A", "B", "C"]
    assert total_count_page2 == 3

    # Test pagination - verify pages don't overlap
    assert not set(unique_values_page1).intersection(set(unique_values_page2))

    # Test pagination - verify all values are covered
    all_values = set(unique_values_page1) | set(unique_values_page2)
    assert all_values == {"A", "B", "C"}

    # Test pagination - size larger than total count
    unique_values_large, total_large = document_store.get_metadata_field_unique_values("meta.category", None, 0, 100)
    assert len(unique_values_large) == 3
    assert set(unique_values_large) == {"A", "B", "C"}
    assert total_large == 3

    # Test pagination - from_ beyond total count (should return empty)
    unique_values_beyond, total_beyond = document_store.get_metadata_field_unique_values("meta.category", None, 10, 10)
    assert len(unique_values_beyond) == 0
    assert total_beyond == 3

    # Test pagination - single item per page
    unique_values_single1, _ = document_store.get_metadata_field_unique_values("meta.category", None, 0, 1)
    unique_values_single2, _ = document_store.get_metadata_field_unique_values("meta.category", None, 1, 1)
    unique_values_single3, _ = document_store.get_metadata_field_unique_values("meta.category", None, 2, 1)
    assert len(unique_values_single1) == 1
    assert len(unique_values_single2) == 1
    assert len(unique_values_single3) == 1
    # All three pages should be different
    assert len(set(unique_values_single1 + unique_values_single2 + unique_values_single3)) == 3

    # Test with search term - filter by content matching "Python"
    unique_values_filtered, total_filtered = document_store.get_metadata_field_unique_values(
        "meta.category", "Python", 0, 10
    )
    assert set(unique_values_filtered) == {"A"}  # Only category A has documents with "Python" in content
    assert total_filtered == 1

    # Test with search term - filter by content matching "Java"
    unique_values_java, total_java = document_store.get_metadata_field_unique_values("meta.category", "Java", 0, 10)
    assert set(unique_values_java) == {"B"}  # Only category B has documents with "Java" in content
    assert total_java == 1

    # Test pagination with search term
    unique_values_search_page1, total_search = document_store.get_metadata_field_unique_values(
        "meta.language", "Python", 0, 1
    )
    assert len(unique_values_search_page1) == 1
    assert unique_values_search_page1[0] == "Python"
    assert total_search == 1

    # Test pagination with search term - beyond results
    unique_values_search_empty, total_search_empty = document_store.get_metadata_field_unique_values(
        "meta.language", "Python", 10, 10
    )
    assert len(unique_values_search_empty) == 0
    assert total_search_empty == 1

    # Test with integer values
    int_docs = [
        Document(content="Doc 1", meta={"priority": 1}),
        Document(content="Doc 2", meta={"priority": 2}),
        Document(content="Doc 3", meta={"priority": 1}),
        Document(content="Doc 4", meta={"priority": 3}),
    ]
    document_store.write_documents(int_docs)
    unique_priorities, total_priorities = document_store.get_metadata_field_unique_values("meta.priority", None, 0, 10)
    assert set(unique_priorities) == {"1", "2", "3"}
    assert total_priorities == 3

    # Test with search term on integer field
    unique_priorities_filtered, total_priorities_filtered = document_store.get_metadata_field_unique_values(
        "meta.priority", "Doc 1", 0, 10
    )
    assert set(unique_priorities_filtered) == {"1"}
    assert total_priorities_filtered == 1
