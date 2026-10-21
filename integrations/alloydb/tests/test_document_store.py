# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

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
from psycopg import Connection, Cursor, Error
from psycopg.sql import SQL

from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore


def test_process_unique_values_result_preserves_non_string_types():
    """Non-string metadata values (e.g. ints) are returned in their original type, not stringified.

    Unit test (no live AlloyDB instance needed) since `_process_unique_values_result` is a pure
    static method operating on already-fetched rows.
    """
    count_result = {"total": 2}
    records = [{"value": 1}, {"value": 2}]

    values, total = AlloyDBDocumentStore._process_unique_values_result(count_result, records)

    assert set(values) == {1, 2}
    assert total == 2


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
    def test_get_metadata_field_unique_values_distinct_types(self, document_store: AlloyDBDocumentStore):
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

    def test_close_and_reopen(self, document_store: AlloyDBDocumentStore):
        assert document_store.count_documents() == 0
        assert document_store._connection is not None

        document_store.close()
        assert document_store._connection is None

        assert document_store.count_documents() == 0
        assert document_store._connection is not None

    def test_get_metadata_fields_info_empty_collection(self, document_store: AlloyDBDocumentStore):
        """Returns empty dict when the store has no documents."""
        assert document_store.count_documents() == 0

        fields_info = document_store.get_metadata_fields_info()
        assert fields_info == {}

    def test_get_metadata_field_min_max_empty_collection(self, document_store: AlloyDBDocumentStore):
        """Returns None min/max when the field doesn't exist in the store."""
        assert document_store.count_documents() == 0

        result = document_store.get_metadata_field_min_max("priority")
        assert result["min"] is None
        assert result["max"] is None

    def test_write_documents(self, document_store: AlloyDBDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_blob(self, document_store: AlloyDBDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        document_store.write_documents(docs)

        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_init(monkeypatch):
    monkeypatch.setenv(
        "ALLOYDB_INSTANCE_URI",
        "projects/p/locations/r/clusters/c/instances/i",
    )
    monkeypatch.setenv("ALLOYDB_USER", "my-user")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "my-password")

    document_store = AlloyDBDocumentStore(
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
def test_init_invalid_vector_function():
    with pytest.raises(ValueError, match=r"vector_function must be one of.*"):
        AlloyDBDocumentStore(vector_function="invalid")


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_to_dict(monkeypatch):
    monkeypatch.setenv(
        "ALLOYDB_INSTANCE_URI",
        "projects/p/locations/r/clusters/c/instances/i",
    )
    monkeypatch.setenv("ALLOYDB_USER", "my-user")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "my-password")

    document_store = AlloyDBDocumentStore(
        create_extension=False,
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

    result = document_store.to_dict()
    assert result["type"] == ("haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore")
    params = result["init_parameters"]
    assert params["create_extension"] is False
    assert params["table_name"] == "my_table"
    assert params["embedding_dimension"] == 512
    assert params["vector_function"] == "l2_distance"
    assert params["recreate_table"] is True
    assert params["search_strategy"] == "hnsw"
    assert params["hnsw_recreate_index_if_exists"] is True
    assert params["hnsw_index_creation_kwargs"] == {"m": 32, "ef_construction": 128}
    assert params["hnsw_index_name"] == "my_hnsw_index"
    assert params["hnsw_ef_search"] == 50
    assert params["keyword_index_name"] == "my_keyword_index"
    assert params["instance_uri"]["type"] == "env_var"
    assert params["user"]["type"] == "env_var"
    assert params["password"]["type"] == "env_var"


def test_connection_is_valid_returns_true_on_success():
    mock_connection = Mock(spec=Connection)
    mock_connection.execute.return_value = None

    assert AlloyDBDocumentStore._connection_is_valid(mock_connection) is True


def test_connection_is_valid_returns_false_when_execute_raises():
    mock_connection = Mock(spec=Connection)
    mock_connection.execute.side_effect = Error("connection dropped")

    assert AlloyDBDocumentStore._connection_is_valid(mock_connection) is False


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


def test_normalize_metadata_field_name_strips_meta_prefix():
    assert AlloyDBDocumentStore._normalize_metadata_field_name("meta.author") == "author"
    assert AlloyDBDocumentStore._normalize_metadata_field_name("author") == "author"


def test_normalize_metadata_field_name_rejects_invalid_characters():
    with pytest.raises(ValueError, match="Invalid metadata field name"):
        AlloyDBDocumentStore._normalize_metadata_field_name("field; DROP TABLE")


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


def test_init_is_lazy(monkeypatch):
    """Constructing the store must not open a database connection."""
    monkeypatch.setenv("ALLOYDB_INSTANCE_URI", "projects/p/locations/r/clusters/c/instances/i")
    monkeypatch.setenv("ALLOYDB_USER", "my-user")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "my-password")

    store = AlloyDBDocumentStore()

    assert store._connection is None
    assert store._cursor is None
    assert store._dict_cursor is None
    assert store._connector is None


INSTANCE_URI = "projects/p/locations/r/clusters/c/instances/i"


@pytest.fixture
def alloydb_env(monkeypatch):
    monkeypatch.setenv("ALLOYDB_INSTANCE_URI", INSTANCE_URI)
    monkeypatch.setenv("ALLOYDB_USER", "my-user")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "my-password")


@pytest.mark.usefixtures("alloydb_env")
def test_create_connection_uses_alloydb_connector():
    store = AlloyDBDocumentStore(db="my-db", ip_type="PUBLIC")

    with patch("google.cloud.alloydbconnector.Connector") as mock_connector_cls:
        connection = store._create_connection()
        store._create_connection()

    # the connector is created once and reused for later connections
    mock_connector_cls.assert_called_once_with(ip_type="PUBLIC")
    mock_connector = mock_connector_cls.return_value
    assert connection is mock_connector.connect.return_value
    assert mock_connector.connect.call_count == 2
    mock_connector.connect.assert_called_with(
        INSTANCE_URI, driver="psycopg", user="my-user", db="my-db", enable_iam_auth=False, password="my-password"
    )


@pytest.mark.usefixtures("alloydb_env")
def test_create_connection_with_iam_auth_omits_password():
    store = AlloyDBDocumentStore(enable_iam_auth=True)

    with patch("google.cloud.alloydbconnector.Connector") as mock_connector_cls:
        store._create_connection()

    mock_connector_cls.return_value.connect.assert_called_once_with(
        INSTANCE_URI, driver="psycopg", user="my-user", db="postgres", enable_iam_auth=True
    )


@pytest.mark.usefixtures("alloydb_env")
def test_create_connection_raises_document_store_error_when_connector_fails():
    store = AlloyDBDocumentStore()

    with patch("google.cloud.alloydbconnector.Connector") as mock_connector_cls:
        mock_connector_cls.return_value.connect.side_effect = RuntimeError("unauthorized")
        with pytest.raises(DocumentStoreError, match="Failed to connect to AlloyDB instance"):
            store._create_connection()


@pytest.mark.usefixtures("alloydb_env")
def test_ensure_db_setup_uses_connection_from_alloydb_connector():
    store = AlloyDBDocumentStore()

    with (
        patch("google.cloud.alloydbconnector.Connector") as mock_connector_cls,
        patch("haystack_integrations.document_stores.pgvector.document_store.register_vector") as mock_register,
        patch.object(AlloyDBDocumentStore, "_initialize_table") as mock_initialize_table,
    ):
        store._ensure_db_setup()

    mock_connection = mock_connector_cls.return_value.connect.return_value
    assert store._connection is mock_connection
    assert mock_connection.autocommit is True
    mock_connection.execute.assert_called_once_with("CREATE EXTENSION IF NOT EXISTS vector")
    mock_register.assert_called_once_with(mock_connection)
    assert store._cursor is not None
    assert store._dict_cursor is not None
    mock_initialize_table.assert_called_once()


def test_close_closes_connector(mock_store):
    mock_connector = Mock()
    mock_store._connector = mock_connector
    mock_store._table_initialized = True

    mock_store.close()

    mock_connector.close.assert_called_once()
    assert mock_store._connector is None
    assert mock_store._table_initialized is False


@pytest.mark.asyncio
async def test_async_methods_are_not_supported(mock_store):
    with pytest.raises(NotImplementedError, match="does not support async operations"):
        await mock_store.count_documents_async()
