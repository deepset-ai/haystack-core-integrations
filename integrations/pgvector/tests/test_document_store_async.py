# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import psycopg
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)
from haystack.utils import Secret
from psycopg import AsyncConnection, Error
from psycopg.cursor_async import AsyncCursor
from psycopg.sql import SQL

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentStoreAsync(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        Embeddings lose float32 precision when round-tripped through pgvector, so we
        compare them approximately and then do an exact equality check on the rest.
        """
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)
            received_doc.embedding, expected_doc.embedding = None, None
            assert received_doc == expected_doc

    async def test_write_documents_async(self, document_store: PgvectorDocumentStore):
        """pgvector default policy raises DuplicateDocumentError on duplicate writes."""
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    async def test_count_not_empty_async(self, document_store: PgvectorDocumentStore):
        """Override: mixin method is missing 'self', causing fixture injection to fail."""
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    async def test_write_blob(self, document_store: PgvectorDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        await document_store.write_documents_async(docs)

        retrieved_docs = await document_store.filter_documents_async()
        assert retrieved_docs == docs

    async def test_connection_check_and_recreation(self, document_store: PgvectorDocumentStore):
        await document_store._ensure_db_setup_async()
        original_connection = document_store._async_connection

        with patch.object(PgvectorDocumentStore, "_connection_is_valid_async", return_value=False):
            await document_store._ensure_db_setup_async()
            new_connection = document_store._async_connection

        # verify that a new connection is created
        assert new_connection is not original_connection
        assert document_store._async_connection == new_connection
        assert original_connection.closed

        assert document_store._async_cursor is not None
        assert document_store._async_dict_cursor is not None

        # test with new connection
        with patch.object(PgvectorDocumentStore, "_connection_is_valid_async", return_value=True):
            await document_store._ensure_db_setup_async()
            same_connection = document_store._async_connection
            assert same_connection is document_store._async_connection

    async def test_invalid_connection_string(self, monkeypatch):
        monkeypatch.setenv("PG_CONN_STR", "invalid_connection_string")
        document_store = PgvectorDocumentStore()
        with pytest.raises(DocumentStoreError) as e:
            await document_store._ensure_db_setup_async()
        assert "Failed to connect to PostgreSQL database" in str(e)

    async def test_update_by_filter_async_empty_meta_raises_error(self, document_store: PgvectorDocumentStore):
        docs = [Document(content="Doc 1", meta={"category": "A"})]
        await document_store.write_documents_async(docs)

        with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
            await document_store.update_by_filter_async(
                filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={}
            )

    async def test_delete_table_async_first_call(self, document_store: PgvectorDocumentStore):
        """
        Test that delete_table_async can be executed as the initial operation on the Document Store
        without triggering errors due to an uninitialized state.
        """
        await document_store.delete_table_async()  # if throw error, test fails


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hnsw_index_recreation():
    async def get_index_oid(cursor, document_store, schema_name, index_name):
        sql_get_index_oid = SQL("""
            SELECT c.oid
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'i'
            AND n.nspname = %s
            AND c.relname = %s;
        """)
        result = await document_store._execute_sql_async(
            cursor=cursor, sql_query=sql_get_index_oid, params=(schema_name, index_name)
        )
        return (await result.fetchone())[0]

    # create a new schema
    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    schema_name = "test_schema"
    async with await psycopg.AsyncConnection.connect(connection_string, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

    # create a first document store and trigger the creation of the hnsw index
    params = {
        "connection_string": Secret.from_token(connection_string),
        "schema_name": schema_name,
        "table_name": "haystack_test_hnsw_index_recreation",
        "search_strategy": "hnsw",
    }
    ds1 = PgvectorDocumentStore(**params)
    await ds1._ensure_db_setup_async()

    # get the hnsw index oid
    hnws_index_name = "haystack_hnsw_index"
    first_oid = await get_index_oid(
        cursor=ds1._async_cursor, document_store=ds1, schema_name=ds1.schema_name, index_name=hnws_index_name
    )

    # create second document store with recreation enabled
    ds2 = PgvectorDocumentStore(**params, hnsw_recreate_index_if_exists=True)
    await ds2._ensure_db_setup_async()

    # get the index oid
    second_oid = await get_index_oid(
        cursor=ds2._async_cursor, document_store=ds2, schema_name=ds2.schema_name, index_name=hnws_index_name
    )

    # verify that oids differ
    assert second_oid != first_oid, "Index was not recreated (OID remained the same)"

    # Clean up: drop the schema after the test
    async with await psycopg.AsyncConnection.connect(connection_string, autocommit=True) as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_table_if_not_exists():
    async def get_table_oid(cursor, document_store, schema_name, table_name):
        sql_get_table_oid = SQL("""
            SELECT c.oid
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
            AND n.nspname = %s
            AND c.relname = %s;
        """)
        result = await document_store._execute_sql_async(
            cursor=cursor, sql_query=sql_get_table_oid, params=(schema_name, table_name)
        )
        return (await result.fetchone())[0]

    connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
    schema_name = "test_schema"
    table_name = "test_table"

    # Create a new schema
    async with await psycopg.AsyncConnection.connect(connection_string, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema_name}")

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(connection_string),
        schema_name=schema_name,
        table_name=table_name,
    )

    await document_store._ensure_db_setup_async()
    await document_store._initialize_table_async()

    first_table_oid = await get_table_oid(
        cursor=document_store._async_cursor,
        document_store=document_store,
        schema_name=schema_name,
        table_name=table_name,
    )
    assert first_table_oid is not None, "Table was not created"

    await document_store._initialize_table_async()
    second_table_oid = await get_table_oid(
        cursor=document_store._async_cursor,
        document_store=document_store,
        schema_name=schema_name,
        table_name=table_name,
    )

    assert first_table_oid == second_table_oid, "Table was recreated"

    # Clean up: drop the schema after the test
    async with await psycopg.AsyncConnection.connect(connection_string, autocommit=True) as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")


@pytest.mark.asyncio
async def test_connection_is_valid_async_returns_true_on_success():
    mock_connection = Mock(spec=AsyncConnection)
    mock_connection.execute.return_value = None

    assert await PgvectorDocumentStore._connection_is_valid_async(mock_connection) is True


@pytest.mark.asyncio
async def test_connection_is_valid_async_returns_false_when_execute_raises():
    mock_connection = Mock(spec=AsyncConnection)
    mock_connection.execute.side_effect = Error("connection dropped")

    assert await PgvectorDocumentStore._connection_is_valid_async(mock_connection) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursor",
    [None, Mock(spec=AsyncCursor)],
    ids=["cursor_is_none", "connection_is_none"],
)
async def test_execute_sql_async_raises_when_not_initialized(mock_store, cursor):
    assert mock_store._async_connection is None
    with pytest.raises(ValueError, match="cursor or the connection is not initialized"):
        await mock_store._execute_sql_async(cursor=cursor, sql_query=SQL("SELECT 1"))


@pytest.mark.asyncio
async def test_execute_sql_async_converts_psycopg_error_to_document_store_error(
    mock_store_with_mock_async_connection,
):
    mock_cursor = Mock(spec=AsyncCursor)
    original_error = Error("connection broken")
    mock_cursor.execute.side_effect = original_error

    with pytest.raises(DocumentStoreError, match="write failed") as exc_info:
        await mock_store_with_mock_async_connection._execute_sql_async(
            cursor=mock_cursor,
            sql_query=SQL("SELECT 1"),
            error_msg="write failed",
        )

    mock_store_with_mock_async_connection._async_connection.rollback.assert_awaited_once_with()
    assert exc_info.value.__cause__ is original_error


@pytest.mark.asyncio
async def test_write_documents_async_rejects_non_document_items(mock_store):
    with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
        await mock_store.write_documents_async([{"not": "a document"}])


@pytest.mark.asyncio
async def test_count_unique_metadata_by_filter_async_rejects_empty_fields(mock_store):
    with pytest.raises(ValueError, match="metadata_fields must be a non-empty list"):
        await mock_store.count_unique_metadata_by_filter_async(filters={}, metadata_fields=[])
