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

        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

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

        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}
        )
        assert deleted_count == 0
        assert await document_store.count_documents_async() == 2

    async def test_delete_by_filter_async_advanced_filters(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "A", "year": 2024, "status": "published"}),
            Document(content="Doc 3", meta={"category": "B", "year": 2023, "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # AND condition
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

        # OR condition
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

        # update multiple fields for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            meta={"status": "published", "priority": "high", "reviewed": True},
        )
        assert updated_count == 2

        # verify
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["status"] == "published"
            assert doc.meta["priority"] == "high"
            assert doc.meta["reviewed"] is True
            assert doc.meta["year"] == 2023  # Original field should still be present

        # verify category B was not updated
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

        # update documents with category="C" (no matches)
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}, meta={"status": "published"}
        )
        assert updated_count == 0
        assert await document_store.count_documents_async() == 2

        # verify no documents were updated
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 0

    async def test_update_by_filter_async_advanced_filters(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "A", "year": 2024, "status": "draft"}),
            Document(content="Doc 3", meta={"category": "B", "year": 2023, "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # AND condition
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

        # verify
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 1
        assert published_docs[0].meta["category"] == "A"
        assert published_docs[0].meta["year"] == 2023

        # OR condition
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

        featured_docs = await document_store.filter_documents_async(
            filters={"field": "meta.featured", "operator": "==", "value": True}
        )
        assert len(featured_docs) == 2

    async def test_update_by_filter_async_empty_meta_raises_error(self, document_store: PgvectorDocumentStore):
        docs = [Document(content="Doc 1", meta={"category": "A"})]
        await document_store.write_documents_async(docs)

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_count_documents_by_filter_async(document_store: PgvectorDocumentStore):
    filterable_docs = [
        Document(content="Doc 1", meta={"category": "A", "status": "active"}),
        Document(content="Doc 2", meta={"category": "B", "status": "active"}),
        Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
        Document(content="Doc 4", meta={"category": "A", "status": "active"}),
    ]
    await document_store.write_documents_async(filterable_docs)
    assert await document_store.count_documents_async() == 4

    count_a = await document_store.count_documents_by_filter_async(
        filters={"field": "meta.category", "operator": "==", "value": "A"}
    )
    assert count_a == 3

    count_active = await document_store.count_documents_by_filter_async(
        filters={"field": "meta.status", "operator": "==", "value": "active"}
    )
    assert count_active == 3

    count_a_active = await document_store.count_documents_by_filter_async(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "A"},
                {"field": "meta.status", "operator": "==", "value": "active"},
            ],
        }
    )
    assert count_a_active == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_count_unique_metadata_by_filter_async(document_store: PgvectorDocumentStore):
    filterable_docs = [
        Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
        Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
        Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
    ]
    await document_store.write_documents_async(filterable_docs)
    assert await document_store.count_documents_async() == 5

    # count distinct values for all documents
    distinct_counts = await document_store.count_unique_metadata_by_filter_async(
        filters={}, metadata_fields=["category", "status", "priority"]
    )
    assert distinct_counts["category"] == 3  # A, B, C
    assert distinct_counts["status"] == 2  # active, inactive
    assert distinct_counts["priority"] == 3  # 1, 2, 3

    # count distinct values for documents with category="A"
    distinct_counts_a = await document_store.count_unique_metadata_by_filter_async(
        filters={"field": "meta.category", "operator": "==", "value": "A"},
        metadata_fields=["category", "status", "priority"],
    )
    assert distinct_counts_a["category"] == 1  # Only A
    assert distinct_counts_a["status"] == 2  # active, inactive
    assert distinct_counts_a["priority"] == 2  # 1, 3

    # count distinct values for documents with status="active"
    distinct_counts_active = await document_store.count_unique_metadata_by_filter_async(
        filters={"field": "meta.status", "operator": "==", "value": "active"},
        metadata_fields=["category", "status", "priority"],
    )
    assert distinct_counts_active["category"] == 3  # A, B, C
    assert distinct_counts_active["status"] == 1  # Only active
    assert distinct_counts_active["priority"] == 3  # 1, 2, 3

    # count distinct values with complex filter (category="A" AND status="active")
    distinct_counts_a_active = await document_store.count_unique_metadata_by_filter_async(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "A"},
                {"field": "meta.status", "operator": "==", "value": "active"},
            ],
        },
        metadata_fields=["category", "status", "priority"],
    )
    assert distinct_counts_a_active["category"] == 1  # Only A
    assert distinct_counts_a_active["status"] == 1  # Only active
    assert distinct_counts_a_active["priority"] == 2  # 1, 3

    # Test with only a subset of fields
    distinct_counts_subset = await document_store.count_unique_metadata_by_filter_async(
        filters={}, metadata_fields=["category", "status"]
    )
    assert distinct_counts_subset["category"] == 3
    assert distinct_counts_subset["status"] == 2
    assert "priority" not in distinct_counts_subset

    # Test field name normalization (with "meta." prefix)
    distinct_counts_normalized = await document_store.count_unique_metadata_by_filter_async(
        filters={}, metadata_fields=["meta.category", "status", "meta.priority"]
    )
    assert distinct_counts_normalized["category"] == 3
    assert distinct_counts_normalized["status"] == 2
    assert distinct_counts_normalized["priority"] == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_metadata_fields_info_async(document_store: PgvectorDocumentStore):
    filterable_docs = [
        Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Doc 2", meta={"category": "B", "status": "inactive"}),
    ]
    await document_store.write_documents_async(filterable_docs)

    fields_info = await document_store.get_metadata_fields_info_async()

    # Verify that fields_info contains expected fields
    assert "content" in fields_info
    assert "category" in fields_info
    assert "status" in fields_info
    assert "priority" in fields_info

    assert fields_info["content"]["type"] == "text"
    assert fields_info["category"]["type"] == "text"
    assert fields_info["status"]["type"] == "text"
    assert fields_info["priority"]["type"] == "integer"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_metadata_field_min_max_async(document_store: PgvectorDocumentStore):
    # Test with integer values
    docs = [
        Document(content="Doc 1", meta={"priority": 1, "age": 10}),
        Document(content="Doc 2", meta={"priority": 5, "age": 20}),
        Document(content="Doc 3", meta={"priority": 3, "age": 15}),
        Document(content="Doc 4", meta={"priority": 10, "age": 5}),
        Document(content="Doc 6", meta={"rating": 10.5}),
        Document(content="Doc 7", meta={"rating": 20.3}),
        Document(content="Doc 8", meta={"rating": 15.7}),
        Document(content="Doc 9", meta={"rating": 5.2}),
    ]
    await document_store.write_documents_async(docs)

    # Test with "meta." prefix for integer field
    min_max_priority = await document_store.get_metadata_field_min_max_async("meta.priority")
    assert min_max_priority["min"] == 1
    assert min_max_priority["max"] == 10

    # Test with "meta." prefix for another integer field
    min_max_age = await document_store.get_metadata_field_min_max_async("meta.age")
    assert min_max_age["min"] == 5
    assert min_max_age["max"] == 20

    # Test with single value
    single_doc = [Document(content="Doc 5", meta={"single_value": 42})]
    await document_store.write_documents_async(single_doc)
    min_max_single = await document_store.get_metadata_field_min_max_async("meta.single_value")
    assert min_max_single["min"] == 42
    assert min_max_single["max"] == 42

    # Test with float values
    min_max_rating = await document_store.get_metadata_field_min_max_async("meta.rating")
    assert min_max_rating["min"] == pytest.approx(5.2)
    assert min_max_rating["max"] == pytest.approx(20.3)

    # Test with text/string values - lexicographic comparison
    text_docs = [
        Document(content="Doc 1", meta={"category": "Zebra", "status": "active"}),
        Document(content="Doc 2", meta={"category": "Apple", "status": "pending"}),
        Document(content="Doc 3", meta={"category": "Banana", "status": "inactive"}),
        Document(content="Doc 4", meta={"category": "apple", "status": "active"}),
    ]
    await document_store.write_documents_async(text_docs)

    # Test lexicographic min/max for text fields (case-sensitive)
    min_max_category = await document_store.get_metadata_field_min_max_async("meta.category")
    assert min_max_category["min"] == "Apple"  # 'A' comes before 'B' and 'Z' and 'a'
    assert min_max_category["max"] == "apple"  # 'a' comes after 'A', 'B', 'Z' in ASCII

    min_max_status = await document_store.get_metadata_field_min_max_async("meta.status")
    assert min_max_status["min"] == "active"  # 'a' comes before 'i' and 'p'
    assert min_max_status["max"] == "pending"  # 'p' comes after 'a' and 'i'

    # Test with empty strings
    empty_string_docs = [
        Document(content="Doc 1", meta={"tag": ""}),
        Document(content="Doc 2", meta={"tag": "A"}),
        Document(content="Doc 3", meta={"tag": "B"}),
    ]
    await document_store.write_documents_async(empty_string_docs)
    min_max_tag = await document_store.get_metadata_field_min_max_async("meta.tag")
    assert min_max_tag["min"] == ""  # Empty string is typically minimum
    assert min_max_tag["max"] == "B"  # 'B' is maximum

    # Test with special characters
    special_char_docs = [
        Document(content="Doc 1", meta={"code": "!@#"}),
        Document(content="Doc 2", meta={"code": "$%^"}),
        Document(content="Doc 3", meta={"code": "&*()"}),
    ]
    await document_store.write_documents_async(special_char_docs)
    min_max_code = await document_store.get_metadata_field_min_max_async("meta.code")
    # Special characters have specific ASCII ordering
    assert min_max_code["min"] in ["!@#", "$%^", "&*()"]
    assert min_max_code["max"] in ["!@#", "$%^", "&*()"]

    # Test with Unicode characters
    unicode_docs = [
        Document(content="Doc 1", meta={"name": "Ángel"}),
        Document(content="Doc 2", meta={"name": "Zebra"}),
        Document(content="Doc 3", meta={"name": "Alpha"}),
    ]
    await document_store.write_documents_async(unicode_docs)
    min_max_name = await document_store.get_metadata_field_min_max_async("meta.name")
    # With COLLATE "C", comparison is byte-order based
    # "Alpha" should be minimum (A comes first in ASCII)
    # "Ángel" or "Zebra" will be maximum depending on byte encoding
    assert min_max_name["min"] == "Alpha"  # 'A' comes first in ASCII
    assert min_max_name["max"] in ["Ángel", "Zebra"]  # Depends on UTF-8 byte encoding


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_metadata_field_unique_values_async(document_store: PgvectorDocumentStore):
    # Test with string values
    docs = [
        Document(content="Python programming", meta={"category": "A", "language": "Python"}),
        Document(content="Java programming", meta={"category": "B", "language": "Java"}),
        Document(content="Python scripting", meta={"category": "A", "language": "Python"}),
        Document(content="JavaScript development", meta={"category": "C", "language": "JavaScript"}),
        Document(content="Python data science", meta={"category": "A", "language": "Python"}),
        Document(content="Java backend", meta={"category": "B", "language": "Java"}),
    ]
    await document_store.write_documents_async(docs)

    # Test getting all unique values without search term
    unique_values, total_count = await document_store.get_metadata_field_unique_values_async(
        "meta.category", None, 0, 10
    )
    assert set(unique_values) == {"A", "B", "C"}
    assert total_count == 3

    # Test with "meta." prefix
    unique_languages, total_languages = await document_store.get_metadata_field_unique_values_async(
        "meta.language", None, 0, 10
    )
    assert set(unique_languages) == {"Python", "Java", "JavaScript"}
    assert total_languages == 3

    # Test pagination - first page
    unique_values_page1, total_count_page1 = await document_store.get_metadata_field_unique_values_async(
        "meta.category", None, 0, 2
    )
    assert len(unique_values_page1) == 2
    assert all(val in ["A", "B", "C"] for val in unique_values_page1)
    assert total_count_page1 == 3

    # Test pagination - second page
    unique_values_page2, total_count_page2 = await document_store.get_metadata_field_unique_values_async(
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
    unique_values_large, total_large = await document_store.get_metadata_field_unique_values_async(
        "meta.category", None, 0, 100
    )
    assert len(unique_values_large) == 3
    assert set(unique_values_large) == {"A", "B", "C"}
    assert total_large == 3

    # Test pagination - from_ beyond total count (should return empty)
    unique_values_beyond, total_beyond = await document_store.get_metadata_field_unique_values_async(
        "meta.category", None, 10, 10
    )
    assert len(unique_values_beyond) == 0
    assert total_beyond == 3

    # Test pagination - single item per page
    unique_values_single1, _ = await document_store.get_metadata_field_unique_values_async("meta.category", None, 0, 1)
    unique_values_single2, _ = await document_store.get_metadata_field_unique_values_async("meta.category", None, 1, 1)
    unique_values_single3, _ = await document_store.get_metadata_field_unique_values_async("meta.category", None, 2, 1)
    assert len(unique_values_single1) == 1
    assert len(unique_values_single2) == 1
    assert len(unique_values_single3) == 1
    # All three pages should be different
    assert len(set(unique_values_single1 + unique_values_single2 + unique_values_single3)) == 3

    # Test with search term - filter by content matching "Python"
    unique_values_filtered, total_filtered = await document_store.get_metadata_field_unique_values_async(
        "meta.category", "Python", 0, 10
    )
    assert set(unique_values_filtered) == {"A"}  # Only category A has documents with "Python" in content
    assert total_filtered == 1

    # Test with search term - filter by content matching "Java"
    unique_values_java, total_java = await document_store.get_metadata_field_unique_values_async(
        "meta.category", "Java", 0, 10
    )
    assert set(unique_values_java) == {"B"}  # Only category B has documents with "Java" in content
    assert total_java == 1

    # Test pagination with search term
    unique_values_search_page1, total_search = await document_store.get_metadata_field_unique_values_async(
        "meta.language", "Python", 0, 1
    )
    assert len(unique_values_search_page1) == 1
    assert unique_values_search_page1[0] == "Python"
    assert total_search == 1

    # Test pagination with search term - beyond results
    unique_values_search_empty, total_search_empty = await document_store.get_metadata_field_unique_values_async(
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
    await document_store.write_documents_async(int_docs)
    unique_priorities, total_priorities = await document_store.get_metadata_field_unique_values_async(
        "meta.priority", None, 0, 10
    )
    assert set(unique_priorities) == {"1", "2", "3"}
    assert total_priorities == 3

    # Test with search term on integer field
    unique_priorities_filtered, total_priorities_filtered = await document_store.get_metadata_field_unique_values_async(
        "meta.priority", "Doc 1", 0, 10
    )
    assert set(unique_priorities_filtered) == {"1"}
    assert total_priorities_filtered == 1
