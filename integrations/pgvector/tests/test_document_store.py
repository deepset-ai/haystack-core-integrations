# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
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
from psycopg.sql import SQL, Composed, Identifier

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.document_stores.pgvector.table_replacement import (
    _prepare_sync_store,
    _Swap,
    _truncate_identifier,
    _validate_identifier,
    _validate_stores,
)


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

    def test_write_documents_counts_every_document(self, document_store: PgvectorDocumentStore):
        docs = [Document(id="1"), Document(id="2"), Document(id="3")]
        assert document_store.write_documents(docs) == 3

    def test_write_documents_skip_counts_only_new_documents(self, document_store: PgvectorDocumentStore):
        document_store.write_documents([Document(id="1")])
        docs = [Document(id="1"), Document(id="2"), Document(id="3")]
        assert document_store.write_documents(docs, DuplicatePolicy.SKIP) == 2

    def test_write_documents_overwrite_counts_every_document(self, document_store: PgvectorDocumentStore):
        document_store.write_documents([Document(id="1", content="old")])
        docs = [Document(id="1", content="new"), Document(id="2"), Document(id="3")]
        assert document_store.write_documents(docs, DuplicatePolicy.OVERWRITE) == 3

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


def test_delete_by_filter_rejects_an_empty_filter(mock_store):
    """An empty filter would compile to an unqualified DELETE and empty the table."""
    with pytest.raises(FilterError, match="non-empty filter"):
        mock_store.delete_by_filter({})


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
def test_build_hnsw_queries_passes_creation_kwargs_as_literals():
    document_store = PgvectorDocumentStore(
        search_strategy="hnsw",
        hnsw_index_creation_kwargs={"m": 32, "ef_construction": "64); DROP TABLE haystack; --", "unknown": 1},
    )

    *_, sql_create_hnsw_index = document_store._build_hnsw_queries()

    assert (
        "SQL(' WITH ('), Composed([Composed([SQL('m'), SQL(' = '), Literal(32)]), SQL(', '), "
        "Composed([SQL('ef_construction'), SQL(' = '), Literal('64); DROP TABLE haystack; --')])]), SQL(')')"
    ) in repr(sql_create_hnsw_index)


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
            "Composed([SQL('1 - (embedding <=> '), Literal('[0.1,2.0]'), SQL(') AS score')])",
            "Composed([SQL('embedding <=> '), Literal('[0.1,2.0]')])",
        ),
        (
            "inner_product",
            "Composed([SQL('(embedding <#> '), Literal('[0.1,2.0]'), SQL(') * -1 AS score')])",
            "Composed([SQL('embedding <#> '), Literal('[0.1,2.0]')])",
        ),
        (
            "l2_distance",
            "Composed([SQL('embedding <-> '), Literal('[0.1,2.0]'), SQL(' AS score')])",
            "Composed([SQL('embedding <-> '), Literal('[0.1,2.0]')])",
        ),
    ],
)
def test_check_and_build_embedding_retrieval_query_orders_by_distance_operator(
    mock_store, vector_function, score_sql, order_by_sql
):
    mock_store.embedding_dimension = 2

    sql_query, params = mock_store._check_and_build_embedding_retrieval_query(
        query_embedding=[0.1, 2],
        vector_function=vector_function,
        top_k=5,
    )

    rendered = repr(sql_query)
    assert f"SQL('SELECT *, '), {score_sql}" in rendered
    assert f"SQL(' ORDER BY '), {order_by_sql}, SQL(' ASC LIMIT '), Literal(5)" in rendered
    assert params == ()


@pytest.mark.parametrize("bad_element", ["0.1]'); DROP TABLE haystack; --", None])
def test_check_and_build_embedding_retrieval_query_rejects_non_numeric_elements(mock_store, bad_element):
    mock_store.embedding_dimension = 2

    with pytest.raises(TypeError, match="query_embedding must be a list of floats"):
        mock_store._check_and_build_embedding_retrieval_query(
            query_embedding=[bad_element, 0.2],
            vector_function="cosine_similarity",
            top_k=5,
        )


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


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_ensure_db_setup_uses_overridden_create_connection(monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "some-connection-string")
    mock_connection = Mock(spec=Connection)

    class CustomConnectionDocumentStore(PgvectorDocumentStore):
        def _create_connection(self) -> Connection:
            return mock_connection

    store = CustomConnectionDocumentStore()
    with patch.object(CustomConnectionDocumentStore, "_initialize_table") as mock_initialize_table:
        store._ensure_db_setup()

    assert store._connection is mock_connection
    assert mock_connection.autocommit is True
    mock_connection.execute.assert_called_once_with("CREATE EXTENSION IF NOT EXISTS vector")
    assert store._cursor is not None
    assert store._dict_cursor is not None
    mock_initialize_table.assert_called_once()


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
    def get_index_oid_and_reloptions(document_store, schema_name, index_name):
        sql_get_index_oid_and_reloptions = """
            SELECT c.oid, c.reloptions
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'i'
            AND n.nspname = %s
            AND c.relname = %s;
        """
        return document_store._cursor.execute(sql_get_index_oid_and_reloptions, (schema_name, index_name)).fetchone()

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

    # get the hnsw index oid and options
    hnws_index_name = "haystack_hnsw_index"
    first_oid, first_reloptions = get_index_oid_and_reloptions(ds1, ds1.schema_name, hnws_index_name)
    assert first_reloptions is None

    # create second document store with recreation enabled and custom index creation kwargs
    ds2 = PgvectorDocumentStore(
        **params, hnsw_recreate_index_if_exists=True, hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128}
    )
    ds2._ensure_db_setup()

    # get the index oid and options
    second_oid, second_reloptions = get_index_oid_and_reloptions(ds2, ds2.schema_name, hnws_index_name)

    # verify that oids differ and the recreated index uses the new kwargs
    assert second_oid != first_oid, "Index was not recreated (OID remained the same)"
    assert sorted(second_reloptions) == ["ef_construction=128", "m=32"]

    # Clean up: drop the schema after the test
    with psycopg.connect(connection_string, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")


@pytest.mark.integration
def test_hnsw_index_creation_kwargs_values_are_not_interpreted_as_sql():
    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    schema_name = "test_schema_hnsw_kwargs_injection"
    with psycopg.connect(connection_string, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        conn.execute(f"CREATE SCHEMA {schema_name}")
        conn.execute(f"CREATE TABLE {schema_name}.canary (id int)")

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(connection_string),
        schema_name=schema_name,
        table_name="haystack_test_hnsw_kwargs_injection",
        search_strategy="hnsw",
        hnsw_index_creation_kwargs={"m": f"16); DROP TABLE {schema_name}.canary; --"},
    )

    # the value is passed as a literal, so PostgreSQL rejects it as an invalid value for "m"
    with pytest.raises(DocumentStoreError, match="Could not create HNSW index"):
        document_store._ensure_db_setup()

    with psycopg.connect(connection_string) as conn:
        canary = conn.execute("SELECT to_regclass(%s)", (f"{schema_name}.canary",)).fetchone()[0]
    assert canary is not None

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
def test_delete_documents_with_quotes_in_ids(document_store: PgvectorDocumentStore):
    """
    The ids are passed as a bound parameter, so quotes in them are not interpreted as SQL.
    """
    docs = [
        Document(id="O'Brien", content="quote in id"),
        Document(id="x') OR ('1'='1", content="injection-shaped id"),
        Document(id="1", content="hello"),
        Document(id="2", content="world"),
    ]
    document_store.write_documents(docs)

    document_store.delete_documents(["O'Brien"])
    assert sorted(doc.id for doc in document_store.filter_documents()) == ["1", "2", "x') OR ('1'='1"]

    # only matches the document with this exact id, instead of every document
    document_store.delete_documents(["x') OR ('1'='1"])
    assert sorted(doc.id for doc in document_store.filter_documents()) == ["1", "2"]


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


def _index_names(store: PgvectorDocumentStore, table_name: str) -> set[str]:
    store._ensure_db_setup()
    assert store._connection is not None
    rows = store._connection.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = %s", (store.schema_name, table_name)
    ).fetchall()
    return {name for (name,) in rows}


def _doc_ids(store: PgvectorDocumentStore) -> list[str]:
    return sorted(document.id for document in store.filter_documents())


@pytest.mark.integration
def test_replace_table_hnsw(document_store_w_hnsw_index: PgvectorDocumentStore, make_staging_store):
    live = document_store_w_hnsw_index
    live.write_documents([Document(id="old", content="old", embedding=[0.1] * 768)])
    staging = make_staging_store(live)
    staging.write_documents([Document(id="new", content="new", embedding=[0.1] * 768)])
    staging_keyword_index, staging_hnsw_index = staging.keyword_index_name, staging.hnsw_index_name

    # Short explicit backup name, so that `<backup>_pkey` fits in PostgreSQL's 63-byte identifier limit
    backup = live.replace_table(staging, backup_table_name=f"backup_{uuid4().hex[:8]}")

    assert [document.id for document in live._embedding_retrieval([0.1] * 768, top_k=5)] == ["new"]
    assert _index_names(live, live.table_name) == {
        live.keyword_index_name,
        live.hnsw_index_name,
        f"{live.table_name}_pkey",
    }
    # The staging store now points at the backup and its renamed indexes, none of which keep staging names
    assert staging.table_name == backup
    backup_indexes = _index_names(live, backup)
    assert backup_indexes == {staging.keyword_index_name, staging.hnsw_index_name, f"{backup}_pkey"}
    assert not backup_indexes & {staging_keyword_index, staging_hnsw_index}
    assert _doc_ids(staging) == ["old"]


@pytest.mark.integration
def test_replace_table_repeated_cycles(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="v0", content="v0")])
    suffix = uuid4().hex[:8]
    # The real workflow: rebuild into a staging table with the same names, swap, drop the backup, repeat
    names = {"table_name": f"stage_{suffix}", "keyword_index_name": f"stage_kw_{suffix}"}

    for cycle in (1, 2, 3):
        staging = make_staging_store(document_store, **names)
        staging.write_documents([Document(id=f"v{cycle}", content=f"v{cycle}")])
        backup = document_store.replace_table(staging)

        assert _doc_ids(document_store) == [f"v{cycle}"]
        assert _doc_ids(staging) == [f"v{cycle - 1}"]
        # Primary key and keyword index names stay attached to the live table on every cycle
        assert _index_names(document_store, document_store.table_name) == {
            document_store.keyword_index_name,
            f"{document_store.table_name}_pkey",
        }
        staging.delete_table()
        assert _index_names(document_store, backup) == set()


@pytest.mark.integration
def test_replace_table_seen_by_other_instance(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="old", content="shared word")])
    # Another instance serving queries, e.g. a pipeline in a different process
    reader = PgvectorDocumentStore(
        connection_string=document_store.connection_string,
        table_name=document_store.table_name,
        embedding_dimension=document_store.embedding_dimension,
    )
    # psycopg prepares a statement after five executions, so the reader holds prepared plans for the old table
    for _ in range(6):
        assert _doc_ids(reader) == ["old"]
        assert [document.id for document in reader._keyword_retrieval("shared", top_k=5)] == ["old"]

    staging = make_staging_store(document_store)
    staging.write_documents([Document(id="new", content="shared word")])
    document_store.replace_table(staging)

    try:
        assert _doc_ids(reader) == ["new"]
        assert [document.id for document in reader._keyword_retrieval("shared", top_k=5)] == ["new"]
    finally:
        reader.close()


@pytest.mark.integration
def test_replace_table_lock_timeout(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="old", content="old")])
    staging = make_staging_store(document_store)
    staging.write_documents([Document(id="new", content="new")])
    live_indexes = _index_names(document_store, document_store.table_name)
    staging_indexes = _index_names(document_store, staging.table_name)

    # A long-running reader: an open transaction keeps a lock on the live table
    with psycopg.connect(document_store.connection_string.resolve_value()) as blocker:
        blocker.execute(SQL("SELECT count(*) FROM {table}").format(table=Identifier(document_store.table_name)))
        with pytest.raises(DocumentStoreError, match="lock timeout"):
            document_store.replace_table(staging, lock_timeout_ms=200)
        blocker.rollback()

    assert _doc_ids(document_store) == ["old"]
    assert _doc_ids(staging) == ["new"]
    assert _index_names(document_store, document_store.table_name) == live_indexes
    assert _index_names(document_store, staging.table_name) == staging_indexes


@pytest.mark.integration
def test_replace_table_host_spelling(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="old", content="old")])
    # Same database as the fixture, which connects through "localhost"
    staging = make_staging_store(
        document_store,
        connection_string=Secret.from_token("postgresql://postgres:postgres@127.0.0.1:5432/postgres"),
    )
    staging.write_documents([Document(id="new", content="new")])

    document_store.replace_table(staging)

    assert _doc_ids(document_store) == ["new"]


@pytest.mark.integration
def test_replace_table_fresh_instance(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="old", content="old")])
    staging = make_staging_store(document_store)
    staging.write_documents([Document(id="new", content="new")])
    # An instance that has not touched the database yet, configured to recreate its table on first setup
    fresh = PgvectorDocumentStore(
        connection_string=document_store.connection_string,
        table_name=document_store.table_name,
        embedding_dimension=document_store.embedding_dimension,
        vector_function=document_store.vector_function,
        recreate_table=True,
    )

    fresh.replace_table(staging)
    # Reconnecting must not run table setup again, which would drop the promoted table
    fresh.close()

    try:
        assert _doc_ids(fresh) == ["new"]
    finally:
        fresh.close()


@pytest.mark.integration
def test_replace_table_backup_pk_name_taken(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="old", content="old")])
    staging = make_staging_store(document_store)
    staging.write_documents([Document(id="new", content="new")])
    backup = f"backup_{uuid4().hex[:8]}"
    # A table already holds the name the backup primary key would get
    blocker = make_staging_store(document_store, table_name=f"{backup}_pkey")
    blocker.write_documents([Document(id="x", content="x")])
    live_indexes = _index_names(document_store, document_store.table_name)

    # Checked before any rename, so the error names the conflict instead of surfacing a mid-swap SQL error
    with pytest.raises(DocumentStoreError, match=f"Backup index relation public.{backup}_pkey already exists"):
        document_store.replace_table(staging, backup_table_name=backup)

    assert _doc_ids(document_store) == ["old"]
    assert _doc_ids(staging) == ["new"]
    assert _index_names(document_store, document_store.table_name) == live_indexes


@pytest.mark.integration
def test_replace_table_rolls_back_midway(document_store: PgvectorDocumentStore, make_staging_store, monkeypatch):
    document_store.write_documents([Document(id="old", content="old")])
    staging = make_staging_store(document_store)
    staging.write_documents([Document(id="new", content="new")])
    live_name, staging_name = document_store.table_name, staging.table_name
    live_indexes = _index_names(document_store, live_name)
    staging_indexes = _index_names(document_store, staging_name)
    backup = f"backup_{uuid4().hex[:8]}"

    # PostgreSQL gives no natural way to fail between renames, so inject a failing statement right after the live
    # table was renamed to the backup name: the point where a non-atomic swap would leave no live table at all
    plan = _Swap.plan

    def plan_failing_midway(self, results):
        statements = plan(self, results)
        return [*statements[:-1], SQL("SELECT 1/0"), statements[-1]]

    monkeypatch.setattr(_Swap, "plan", plan_failing_midway)

    with pytest.raises(DocumentStoreError, match="division by zero"):
        document_store.replace_table(staging, backup_table_name=backup)

    assert document_store.table_name == live_name
    assert staging.table_name == staging_name
    assert _doc_ids(document_store) == ["old"]
    assert _doc_ids(staging) == ["new"]
    assert _index_names(document_store, live_name) == live_indexes
    assert _index_names(document_store, staging_name) == staging_indexes
    assert _index_names(document_store, backup) == set()


@pytest.mark.integration
def test_replace_table_missing_staging(document_store: PgvectorDocumentStore, make_staging_store):
    document_store.write_documents([Document(id="old", content="old")])
    # Never written to, so its table does not exist
    staging = make_staging_store(document_store)

    with pytest.raises(DocumentStoreError, match=f"Staging table public.{staging.table_name} does not exist"):
        document_store.replace_table(staging)

    assert _doc_ids(document_store) == ["old"]
    # Connecting the staging store for the swap must not create an empty table that could be promoted
    assert document_store._connection is not None
    created = document_store._connection.execute(
        "SELECT 1 FROM pg_tables WHERE schemaname = %s AND tablename = %s", ("public", staging.table_name)
    ).fetchone()
    assert created is None


# replace_table: unit tests of the validation and planning helpers, which do no I/O


def _store_pair(
    active: dict[str, Any] | None = None, staging: dict[str, Any] | None = None
) -> tuple[PgvectorDocumentStore, PgvectorDocumentStore]:
    active_store = PgvectorDocumentStore(
        **{"table_name": "live", "keyword_index_name": "live_kw", "hnsw_index_name": "live_hnsw", **(active or {})}
    )
    staging_store = PgvectorDocumentStore(
        **{"table_name": "stage", "keyword_index_name": "stage_kw", "hnsw_index_name": "stage_hnsw", **(staging or {})}
    )
    return active_store, staging_store


@pytest.mark.parametrize("identifier", ["a" * 100, "文" * 30])
def test_truncate_identifier_fits_postgres_limit(identifier):
    result = _truncate_identifier(identifier)

    assert len(result.encode("utf-8")) <= 63
    # The prefix is cut on a character boundary, then made unique with a random suffix
    assert identifier.startswith(result.rsplit("_", 1)[0])
    assert _truncate_identifier("a" * 63) == "a" * 63


@pytest.mark.parametrize(
    ("identifier", "match"), [("", "must not be empty"), ("a" * 64, "at most 63"), ("文" * 22, "at most 63")]
)
def test_validate_identifier_rejects_invalid_identifiers(identifier, match):
    with pytest.raises(ValueError, match=match):
        _validate_identifier(identifier, "name")
    _validate_identifier("a" * 63, "name")


@pytest.mark.parametrize("search_strategy", ["exact_nearest_neighbor", "hnsw"])
def test_validate_stores_generates_backup_names(search_strategy):
    active, staging = _store_pair({"search_strategy": search_strategy}, {"search_strategy": search_strategy})

    backup, renames = _validate_stores(active, staging, None, 5000)

    assert backup.startswith("live_backup_")
    expected = [("live_kw", "stage_kw")] + ([("live_hnsw", "stage_hnsw")] if search_strategy == "hnsw" else [])
    assert [(active_name, staging_name) for active_name, staging_name, _ in renames] == expected
    for _, _, backup_index in renames:
        assert backup_index.startswith(f"{backup}_")
        assert len(backup_index.encode("utf-8")) <= 63


@pytest.mark.parametrize(
    ("active", "staging", "backup", "match"),
    [
        ({"schema_name": "other"}, None, None, "same schema"),
        (None, {"table_name": "live"}, None, "different table names"),
        (None, {"embedding_dimension": 1024}, None, "same embedding_dimension"),
        (None, {"vector_type": "halfvec"}, None, "same vector_type"),
        (None, {"vector_function": "inner_product"}, None, "same vector_function"),
        (None, {"language": "simple"}, None, "same language"),
        (None, {"search_strategy": "hnsw"}, None, "same search_strategy"),
        (
            {"search_strategy": "hnsw", "hnsw_index_creation_kwargs": {"m": 16}},
            {"search_strategy": "hnsw", "hnsw_index_creation_kwargs": {"m": 32}},
            None,
            "same hnsw_index_creation_kwargs",
        ),
        (None, {"keyword_index_name": "live_kw"}, None, "keyword indexes must have different names"),
        (
            {"search_strategy": "hnsw"},
            {"search_strategy": "hnsw", "hnsw_index_name": "live_hnsw"},
            None,
            "HNSW indexes must have different names",
        ),
        ({"table_name": "a" * 64}, None, None, "active table_name must be at most 63"),
        (None, {"keyword_index_name": "k" * 64}, None, "staging keyword_index_name must be at most 63"),
        (
            {"search_strategy": "hnsw", "hnsw_index_name": "h" * 64},
            {"search_strategy": "hnsw"},
            None,
            "hnsw_index_name",
        ),
        (None, None, "b" * 64, "backup_table_name must be at most 63"),
        (None, None, "stage", "must differ from the active and staging table names"),
    ],
)
def test_validate_stores_rejects_incompatible_stores(active, staging, backup, match):
    active_store, staging_store = _store_pair(active, staging)

    with pytest.raises(ValueError, match=match):
        _validate_stores(active_store, staging_store, backup, 5000)


def test_validate_stores_rejects_same_instance():
    active, _ = _store_pair()

    with pytest.raises(ValueError, match="different PgvectorDocumentStore instance"):
        _validate_stores(active, active, None, 5000)


@pytest.mark.parametrize(
    ("lock_timeout_ms", "error"),
    [(True, TypeError), (1.5, TypeError), ("1000", TypeError), (None, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_validate_stores_rejects_invalid_lock_timeouts(lock_timeout_ms, error):
    active, staging = _store_pair()

    with pytest.raises(error, match="lock_timeout_ms"):
        _validate_stores(active, staging, None, lock_timeout_ms)


def test_swap_create_truncates_long_backup_primary_key_name():
    active, staging = _store_pair()

    swap = _Swap.create(active, staging, "b" * 62, 5000)

    # "<backup>_pkey" would be 67 bytes, longer than PostgreSQL accepts
    assert len(swap.backup_pk.encode("utf-8")) <= 63
    assert swap.backup_pk.startswith("b" * 50)


@pytest.mark.parametrize(
    ("kinds", "match"),
    [
        ([("stage", "r")], "Active table public.live does not exist"),
        ([("live", "r")], "Staging table public.stage does not exist"),
        ([("live", "r"), ("stage", "v")], "Staging table public.stage does not exist as a regular table"),
    ],
)
def test_swap_check_tables_exist(kinds, match):
    swap = _swap()

    with pytest.raises(DocumentStoreError, match=match):
        swap.check_tables_exist(kinds)
    swap.check_tables_exist([("live", "r"), ("stage", "r")])


def _swap() -> _Swap:
    return _Swap(
        schema="public",
        active_table="live",
        staging_table="stage",
        backup_table="backup",
        index_renames=[("live_kw", "stage_kw", "backup_kw")],
        backup_pk="backup_pkey",
        lock_timeout_ms=5000,
    )


def _catalog(**overrides: Any) -> list[list[tuple]]:
    """Catalog rows in the order of `_Swap.catalog_queries`, for two compatible tables with primary keys."""
    columns = [("id", "character varying(128)", True, "", ""), ("content", "text", False, "", "")]
    rows: dict[str, list[tuple]] = {
        "active_columns": columns,
        "staging_columns": columns,
        "active_constraints": [("p", "PRIMARY KEY (id)")],
        "staging_constraints": [("p", "PRIMARY KEY (id)")],
        "indexes": [
            ("live", "live_kw", False),
            ("live", "live_pkey", True),
            ("stage", "stage_kw", False),
            ("stage", "stage_pkey", True),
        ],
        "existing": [],
        **overrides,
    }
    return list(rows.values())


def _rename(kind: str, old_name: str, new_name: str) -> Composed:
    return SQL(f"ALTER {kind} {{schema}}.{{old_name}} RENAME TO {{new_name}}").format(
        schema=Identifier("public"), old_name=Identifier(old_name), new_name=Identifier(new_name)
    )


def test_swap_plan_frees_active_names_before_renaming_staging_objects():
    assert _swap().plan(_catalog()) == [
        _rename("INDEX", "live_kw", "backup_kw"),
        _rename("INDEX", "live_pkey", "backup_pkey"),
        _rename("INDEX", "stage_kw", "live_kw"),
        _rename("INDEX", "stage_pkey", "live_pkey"),
        _rename("TABLE", "live", "backup"),
        _rename("TABLE", "stage", "live"),
    ]


def test_swap_plan_without_primary_keys():
    catalog = _catalog(
        active_constraints=[],
        staging_constraints=[],
        indexes=[("live", "live_kw", False), ("stage", "stage_kw", False)],
    )

    assert _swap().plan(catalog) == [
        _rename("INDEX", "live_kw", "backup_kw"),
        _rename("INDEX", "stage_kw", "live_kw"),
        _rename("TABLE", "live", "backup"),
        _rename("TABLE", "stage", "live"),
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"staging_columns": [("id", "text", True, "", "")]}, "incompatible column definitions"),
        ({"staging_constraints": []}, "incompatible constraints"),
        ({"existing": [("backup",)]}, "Backup relation public.backup already exists"),
        ({"existing": [("backup_kw",)]}, "Backup index relation public.backup_kw already exists"),
        ({"existing": [("backup_pkey",)]}, "Backup index relation public.backup_pkey already exists"),
        (
            {"indexes": [("live", "live_pkey", True), ("stage", "stage_kw", False), ("stage", "stage_pkey", True)]},
            "Expected index 'live_kw' was not found on active table 'live'",
        ),
        (
            {"indexes": [("live", "live_kw", False), ("live", "live_pkey", True), ("stage", "stage_pkey", True)]},
            "Expected index 'stage_kw' was not found on staging table 'stage'",
        ),
    ],
)
def test_swap_plan_rejects_incompatible_catalog(overrides, match):
    with pytest.raises(DocumentStoreError, match=match):
        _swap().plan(_catalog(**overrides))


@pytest.mark.parametrize("setup_fails", [False, True])
def test_prepare_sync_store_never_recreates_or_initializes_tables(monkeypatch, setup_fails):
    store = PgvectorDocumentStore(recreate_table=True, hnsw_recreate_index_if_exists=True)
    seen = {}

    def ensure_db_setup():
        seen.update(recreate=store.recreate_table, recreate_hnsw=store.hnsw_recreate_index_if_exists)
        seen.update(initialized=store._table_initialized)
        if setup_fails:
            msg = "connection failed"
            raise DocumentStoreError(msg)

    monkeypatch.setattr(store, "_ensure_db_setup", ensure_db_setup)

    with pytest.raises(DocumentStoreError) if setup_fails else nullcontext():
        _prepare_sync_store(store)

    # While connecting, setup must neither drop nor create a table
    assert seen == {"recreate": False, "recreate_hnsw": False, "initialized": True}
    # Afterwards, and also when connecting fails, the caller's settings are restored
    assert (store.recreate_table, store.hnsw_recreate_index_if_exists, store._table_initialized) == (True, True, False)
