# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import psycopg
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from psycopg.sql import SQL

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentStoreAsync:
    async def test_write_documents(self, document_store: PgvectorDocumentStore):
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    async def test_write_blob(self, document_store: PgvectorDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        await document_store.write_documents_async(docs)

        retrieved_docs = await document_store.filter_documents_async()
        assert retrieved_docs == docs

    async def test_count_documents(self, document_store: PgvectorDocumentStore):
        await document_store.write_documents_async(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    async def test_filter_documents(self, document_store: PgvectorDocumentStore):
        filterable_docs = [
            Document(
                content="1",
                meta={
                    "number": -10,
                },
            ),
            Document(
                content="2",
                meta={
                    "number": 100,
                },
            ),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )

        assert result == [d for d in filterable_docs if d.meta.get("number") == 100]

    async def test_delete_documents(self, document_store: PgvectorDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

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

    async def test_delete_all_documents_async(self, document_store: PgvectorDocumentStore) -> None:
        document_store.write_documents([Document(id=str(i)) for i in range(10)])
        await document_store.delete_all_documents_async()
        assert document_store.count_documents() == 0
        document_store.write_documents([Document(id="1")])
        assert document_store.count_documents() == 1

    async def test_invalid_connection_string(self, monkeypatch):
        monkeypatch.setenv("PG_CONN_STR", "invalid_connection_string")
        document_store = PgvectorDocumentStore()
        with pytest.raises(DocumentStoreError) as e:
            await document_store._ensure_db_setup_async()
        assert "Failed to connect to PostgreSQL database" in str(e)

    async def test_delete_by_filter_async(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023}),
            Document(content="Doc 2", meta={"category": "B", "year": 2023}),
            Document(content="Doc 3", meta={"category": "A", "year": 2024}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Delete documents with category="A"
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        # Verify only category B remains
        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

        # Delete remaining document with year filter
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.year", "operator": "==", "value": 2023}
        )
        assert deleted_count == 1
        assert await document_store.count_documents_async() == 0

    async def test_delete_by_filter_async_no_matches(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        # Try to delete documents with category="C" (no matches)
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}
        )
        assert deleted_count == 0
        assert await document_store.count_documents_async() == 2

    async def test_delete_by_filter_async_complex_filters(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "A", "year": 2024, "status": "published"}),
            Document(content="Doc 3", meta={"category": "B", "year": 2023, "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Delete with AND condition
        deleted_count = await document_store.delete_by_filter_async(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.year", "operator": "==", "value": 2023},
                ],
            }
        )
        assert deleted_count == 1
        assert await document_store.count_documents_async() == 2

        # Delete with OR condition
        deleted_count = await document_store.delete_by_filter_async(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "B"},
                    {"field": "meta.status", "operator": "==", "value": "published"},
                ],
            }
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 0

    async def test_update_by_filter_async(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update status for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 2

        # Verify the updates
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

        # Verify category B still has draft status
        draft_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "draft"}
        )
        assert len(draft_docs) == 1
        assert draft_docs[0].meta["category"] == "B"

    async def test_update_by_filter_async_multiple_fields(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023}),
            Document(content="Doc 2", meta={"category": "A", "year": 2023}),
            Document(content="Doc 3", meta={"category": "B", "year": 2024}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update multiple fields for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            meta={"status": "published", "priority": "high", "reviewed": True},
        )
        assert updated_count == 2

        # Verify the updates
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["status"] == "published"
            assert doc.meta["priority"] == "high"
            assert doc.meta["reviewed"] is True
            assert doc.meta["year"] == 2023  # Original field should still be present

        # Verify category B was not updated
        b_docs = await document_store.filter_documents_async(
            filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert len(b_docs) == 1
        assert "status" not in b_docs[0].meta
        assert "priority" not in b_docs[0].meta

    async def test_update_by_filter_async_no_matches(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        # Try to update documents with category="C" (no matches)
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}, meta={"status": "published"}
        )
        assert updated_count == 0
        assert await document_store.count_documents_async() == 2

        # Verify no documents were updated
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 0

    async def test_update_by_filter_async_complex_filters(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "A", "year": 2024, "status": "draft"}),
            Document(content="Doc 3", meta={"category": "B", "year": 2023, "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update with AND condition
        updated_count = await document_store.update_by_filter_async(
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

        # Verify only one document was updated
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 1
        assert published_docs[0].meta["category"] == "A"
        assert published_docs[0].meta["year"] == 2023

        # Update with OR condition
        updated_count = await document_store.update_by_filter_async(
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

        # Verify both documents were updated
        featured_docs = await document_store.filter_documents_async(
            filters={"field": "meta.featured", "operator": "==", "value": True}
        )
        assert len(featured_docs) == 2

    async def test_update_by_filter_async_empty_meta_raises_error(self, document_store: PgvectorDocumentStore):
        docs = [Document(content="Doc 1", meta={"category": "A"})]
        await document_store.write_documents_async(docs)

        # Empty meta dict should raise ValueError
        with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
            await document_store.update_by_filter_async(
                filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={}
            )


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_table_async_first_call(document_store):
    """
    Test that delete_table_async can be executed as the initial operation on the Document Store
    without triggering errors due to an uninitialized state.
    """
    await document_store.delete_table_async()  # if throw error, test fails


