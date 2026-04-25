# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

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

    def test_invalid_connection(self, monkeypatch):
        monkeypatch.setenv(
            "ALLOYDB_INSTANCE_URI",
            "projects/invalid/locations/invalid/clusters/invalid/instances/invalid",
        )
        monkeypatch.setenv("ALLOYDB_USER", "invalid_user")
        monkeypatch.setenv("ALLOYDB_PASSWORD", "invalid_password")

        document_store = AlloyDBDocumentStore()
        with pytest.raises(DocumentStoreError, match="Failed to connect to AlloyDB instance"):
            document_store._ensure_db_setup()


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


def test_close_is_idempotent(mock_store):
    """Calling close() multiple times should not raise."""
    mock_store.close()
    mock_store.close()
