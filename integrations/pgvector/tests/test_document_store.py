# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

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
from psycopg import Connection, Cursor, Error
from psycopg.sql import SQL

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
        """Returns empty dict when the store has no documents."""
        assert document_store.count_documents() == 0

        fields_info = document_store.get_metadata_fields_info()
        assert fields_info == {}

    def test_get_metadata_field_min_max_empty_collection(self, document_store: PgvectorDocumentStore):
        """Returns None min/max when the field doesn't exist in the store."""
        assert document_store.count_documents() == 0

        result = document_store.get_metadata_field_min_max("priority")
        assert result["min"] is None
        assert result["max"] is None

    def test_write_documents(self, document_store: PgvectorDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_get_metadata_field_unique_values_distinct_types(self, document_store: PgvectorDocumentStore):
        """
        Override: the base mixin test stores int, float, str and bool under the *same* metadata field
        name and expects all four back as distinct values. This store's ``meta`` column is JSONB, and
        PostgreSQL's JSONB equality treats a whole-number float (e.g. 1.0) and a numerically equal int
        (1) as the same value, so ``SELECT DISTINCT meta->'field'`` collapses them regardless of which
        other values share that field.

        This adapts the same intent - int, float, str and bool must come back as distinct, unmangled
        types via get_metadata_field_unique_values() - using one field per type instead of one shared
        field, which is what this store can actually support.

        The float value is a non-whole number (1.5, not 1.0): a whole-number float would still collapse
        with an int under PostgreSQL's JSONB numeric equality even in its own field, so a fractional
        value is used to sidestep that ambiguity entirely.
        """
        docs = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        document_store.write_documents(docs)

        int_values, int_count = document_store.get_metadata_field_unique_values(metadata_field="priority_int")
        str_values, str_count = document_store.get_metadata_field_unique_values(metadata_field="priority_str")
        float_values, float_count = document_store.get_metadata_field_unique_values(metadata_field="priority_float")
        bool_values, bool_count = document_store.get_metadata_field_unique_values(metadata_field="priority_bool")

        assert (int_count, str_count, float_count, bool_count) == (1, 1, 1, 1)
        assert int_values == [1] and type(int_values[0]) is int
        assert str_values == ["1"] and type(str_values[0]) is str
        assert float_values == [1.5] and type(float_values[0]) is float
        assert bool_values == [True] and type(bool_values[0]) is bool

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

    def test_close_and_reopen(self, document_store: PgvectorDocumentStore):
        assert document_store.count_documents() == 0
        assert document_store._connection is not None

        document_store.close()
        assert document_store._connection is None
        assert document_store._cursor is None
        assert document_store._dict_cursor is None

        assert document_store.count_documents() == 0
        assert document_store._connection is not None


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
def test_init_invalid_vector_type():
    with pytest.raises(ValueError, match=r"vector_type must be one of.*"):
        PgvectorDocumentStore(vector_type="invalid")


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_init_invalid_vector_function():
    with pytest.raises(ValueError, match=r"vector_function must be one of.*"):
        PgvectorDocumentStore(vector_function="invalid")


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


def test_connection_is_valid_returns_true_on_success():
    mock_connection = Mock(spec=Connection)
    mock_connection.execute.return_value = None

    assert PgvectorDocumentStore._connection_is_valid(mock_connection) is True


def test_connection_is_valid_returns_false_when_execute_raises():
    mock_connection = Mock(spec=Connection)
    mock_connection.execute.side_effect = Error("connection dropped")

    assert PgvectorDocumentStore._connection_is_valid(mock_connection) is False


@pytest.mark.parametrize(
    "cursor",
    [None, Mock(spec=Cursor)],
    ids=["cursor_is_none", "connection_is_none"],
)
def test_execute_sql_raises_when_not_initialized(mock_store, cursor):
    assert mock_store._connection is None
    with pytest.raises(ValueError, match="cursor or the connection is not initialized"):
        mock_store._execute_sql(cursor=cursor, sql_query=SQL("SELECT 1"))


def test_execute_sql_converts_psycopg_error_to_document_store_error(mock_store_with_mock_connection):
    mock_cursor = Mock(spec=Cursor)
    original_error = Error("connection broken")
    mock_cursor.execute.side_effect = original_error

    with pytest.raises(DocumentStoreError, match="write failed") as exc_info:
        mock_store_with_mock_connection._execute_sql(
            cursor=mock_cursor,
            sql_query=SQL("SELECT 1"),
            error_msg="write failed",
        )

    mock_store_with_mock_connection._connection.rollback.assert_called_once_with()
    assert exc_info.value.__cause__ is original_error


def test_write_documents_rejects_non_document_items(mock_store):
    with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
        mock_store.write_documents([{"not": "a document"}])


def test_count_unique_metadata_by_filter_rejects_empty_fields(mock_store):
    with pytest.raises(ValueError, match="metadata_fields must be a non-empty list"):
        mock_store.count_unique_metadata_by_filter(filters={}, metadata_fields=[])


def test_check_and_build_embedding_retrieval_query_rejects_invalid_vector_function(mock_store):
    with pytest.raises(ValueError, match="vector_function must be one of"):
        mock_store._check_and_build_embedding_retrieval_query(
            query_embedding=[0.1] * mock_store.embedding_dimension,
            vector_function="invalid",
            top_k=5,
        )


@pytest.mark.parametrize(
    ("vector_function", "score_sql", "order_by_sql"),
    [
        (
            "cosine_similarity",
            "1 - (embedding <=> '[0.1,0.2]') AS score",
            "embedding <=> '[0.1,0.2]'",
        ),
        (
            "inner_product",
            "(embedding <#> '[0.1,0.2]') * -1 AS score",
            "embedding <#> '[0.1,0.2]'",
        ),
        (
            "l2_distance",
            "embedding <-> '[0.1,0.2]' AS score",
            "embedding <-> '[0.1,0.2]'",
        ),
    ],
)
def test_check_and_build_embedding_retrieval_query_orders_by_distance_operator(
    mock_store, vector_function, score_sql, order_by_sql
):
    mock_store.embedding_dimension = 2

    sql_query, params = mock_store._check_and_build_embedding_retrieval_query(
        query_embedding=[0.1, 0.2],
        vector_function=vector_function,
        top_k=5,
    )

    rendered = repr(sql_query)
    assert score_sql in rendered
    assert "SQL(' ORDER BY ')" in rendered
    assert f'SQL("{order_by_sql}")' in rendered
    assert "SQL(' ASC LIMIT ')" in rendered
    assert "Literal(5)" in rendered
    assert "ORDER BY score" not in rendered
    assert params == ()


@pytest.mark.parametrize(
    "bad_field",
    [
        "field@invalid",
        "field with spaces",
        "field;drop",
        "meta.field!",
        "field' OR '1'='1",
        "../etc/passwd",
        "field/* comment */",
    ],
)
def test_normalize_metadata_field_name_rejects_invalid_chars(bad_field):
    with pytest.raises(ValueError, match="Invalid metadata field name"):
        PgvectorDocumentStore._normalize_metadata_field_name(bad_field)


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (True, "boolean"),
        (42, "integer"),
        (3.14, "real"),
        ("hello", "text"),
        (["list", "fallback"], "text"),
    ],
)
def test_infer_metadata_field_type(value, expected_type):
    assert PgvectorDocumentStore._infer_metadata_field_type(value) == expected_type


def test_analyze_metadata_fields_skips_non_dict_meta():
    records = [{"meta": "not a dict"}, {"meta": None}]
    assert PgvectorDocumentStore._analyze_metadata_fields_from_records(records) == {}


def test_analyze_metadata_fields_defaults_null_first_value_to_text():
    records = [{"meta": {"tag": None}}, {"meta": {"tag": 42}}]
    result = PgvectorDocumentStore._analyze_metadata_fields_from_records(records)
    assert result == {"tag": {"type": "text"}}


@pytest.mark.parametrize(
    "result",
    [None, {"min_value": None, "max_value": None}],
    ids=["result_is_none", "values_are_none"],
)
def test_process_min_max_result_raises_when_no_values(result):
    with pytest.raises(ValueError, match="Metadata field 'priority' has no values"):
        PgvectorDocumentStore._process_min_max_result("priority", result)


def test_process_count_unique_metadata_result_returns_zero_dict_when_result_none():
    counts = PgvectorDocumentStore._process_count_unique_metadata_result(None, ["category", "language"])
    assert counts == {"category": 0, "language": 0}


def test_process_count_unique_metadata_result_uses_zero_for_missing_keys():
    counts = PgvectorDocumentStore._process_count_unique_metadata_result({"category": 5}, ["category", "language"])
    assert counts == {"category": 5, "language": 0}


def test_close(mock_store):
    mock_connection = Mock(spec=Connection)
    mock_store._connection = mock_connection
    mock_store._cursor = Mock(spec=Cursor)
    mock_store._dict_cursor = Mock(spec=Cursor)

    mock_store.close()

    mock_connection.close.assert_called_once()
    assert mock_store._connection is None
    assert mock_store._cursor is None
    assert mock_store._dict_cursor is None

    mock_store.close()
    mock_connection.close.assert_called_once()


def test_close_is_exception_safe(mock_store):
    mock_connection = Mock(spec=Connection)
    mock_connection.close.side_effect = RuntimeError("boom")
    mock_store._connection = mock_connection

    mock_store.close()

    assert mock_store._connection is None


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
def test_delete_documents_empty_list_is_noop(document_store: PgvectorDocumentStore):
    docs = [
        Document(id="1", content="hello"),
        Document(id="2", content="world"),
    ]
    document_store.write_documents(docs)
    before = sorted(document_store.filter_documents(), key=lambda d: d.id)
    assert len(before) == 2

    document_store.delete_documents([])

    after = sorted(document_store.filter_documents(), key=lambda d: d.id)
    assert after == before


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
def test_get_metadata_field_unique_values_pagination_edge_cases(document_store: PgvectorDocumentStore):
    """
    Edge cases not covered by the shared GetMetadataFieldUniqueValuesTest mixin's pagination test:
    size larger than the total count, from_ beyond the total count, and single-item-per-page paging.
    """
    docs = [
        Document(content="Python programming", meta={"category": "A"}),
        Document(content="Java programming", meta={"category": "B"}),
        Document(content="JavaScript development", meta={"category": "C"}),
    ]
    document_store.write_documents(docs)

    unique_values_large, total_large = document_store.get_metadata_field_unique_values("meta.category", None, 0, 100)
    assert set(unique_values_large) == {"A", "B", "C"}
    assert total_large == 3

    unique_values_beyond, total_beyond = document_store.get_metadata_field_unique_values("meta.category", None, 10, 10)
    assert len(unique_values_beyond) == 0
    assert total_beyond == 3

    unique_values_single1, _ = document_store.get_metadata_field_unique_values("meta.category", None, 0, 1)
    unique_values_single2, _ = document_store.get_metadata_field_unique_values("meta.category", None, 1, 1)
    unique_values_single3, _ = document_store.get_metadata_field_unique_values("meta.category", None, 2, 1)
    assert len(unique_values_single1) == 1
    assert len(unique_values_single2) == 1
    assert len(unique_values_single3) == 1
    assert len(set(unique_values_single1 + unique_values_single2 + unique_values_single3)) == 3
