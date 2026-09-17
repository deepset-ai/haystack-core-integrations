# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Async tests for IBM Db2 Document Store."""

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
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

from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore

# ---------------------------------------------------------------------------
# Integration tests — require a live DB2 instance (docker-compose)
# ---------------------------------------------------------------------------


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
    """
    Mixin-driven async integration tests for IBMDb2DocumentStore.

    Each mixin provides a standard suite of async tests from
    ``haystack.testing.document_store_async``.  The ``document_store``
    fixture is provided by ``conftest.py``.
    """

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]) -> None:
        """
        Embeddings lose FLOAT32 precision after a round-trip through Db2 VECTOR columns,
        so compare them with ``pytest.approx`` and do an exact equality check on the rest.
        """
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for recv, exp in zip(received, expected, strict=True):
            if recv.embedding is None:
                assert exp.embedding is None
            else:
                assert recv.embedding == pytest.approx(exp.embedding, abs=1e-5)
            assert dataclasses.replace(recv, embedding=None) == dataclasses.replace(exp, embedding=None)

    # -- Overrides required by Db2-specific behaviour -----------------------

    async def test_write_documents_async(self, document_store: IBMDb2DocumentStore) -> None:
        """Default policy raises DuplicateDocumentError on duplicate writes."""
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    async def test_get_metadata_field_unique_values_distinct_types_async(
        self, document_store: IBMDb2DocumentStore
    ) -> None:
        """
        Db2's BSON/JSON round-trip keeps numeric types distinct; one field per type avoids
        the int/float collision that some backends exhibit.
        """
        docs = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        await document_store.write_documents_async(docs)

        int_vals, int_cnt = await document_store.get_metadata_field_unique_values_async("priority_int")
        str_vals, str_cnt = await document_store.get_metadata_field_unique_values_async("priority_str")
        float_vals, float_cnt = await document_store.get_metadata_field_unique_values_async("priority_float")
        bool_vals, bool_cnt = await document_store.get_metadata_field_unique_values_async("priority_bool")

        assert (int_cnt, str_cnt, float_cnt, bool_cnt) == (1, 1, 1, 1)
        assert int_vals == [1] and type(int_vals[0]) is int
        assert str_vals == ["1"] and type(str_vals[0]) is str
        assert float_vals == [1.5] and type(float_vals[0]) is float
        assert bool_vals == [True] and type(bool_vals[0]) is bool

    async def test_count_not_empty_async(self, document_store: IBMDb2DocumentStore) -> None:
        """Explicit count test with known documents."""
        await document_store.write_documents_async(
            [Document(content="doc 1"), Document(content="doc 2"), Document(content="doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    async def test_update_by_filter_async_empty_meta_raises_error(self, document_store: IBMDb2DocumentStore) -> None:
        docs = [Document(content="Doc A", meta={"category": "A"})]
        await document_store.write_documents_async(docs)
        with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
            await document_store.update_by_filter_async(
                filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={}
            )

    async def test_close_async_and_reopen(self, document_store: IBMDb2DocumentStore) -> None:
        """close_async() clears the async connection; the next call re-establishes it."""
        assert await document_store.count_documents_async() == 0
        assert document_store._async_connection is not None

        await document_store.close_async()
        assert document_store._async_connection is None

        # re-initialize on next use
        assert await document_store.count_documents_async() == 0
        assert document_store._async_connection is not None

    async def test_close_async_does_not_affect_sync_connection(self, document_store: IBMDb2DocumentStore) -> None:
        """close_async() must leave the synchronous connection intact."""
        # Force sync connection to be created
        _ = document_store.count_documents()
        sync_conn = document_store._connection
        assert sync_conn is not None

        await document_store.close_async()

        # Sync connection is still alive
        assert document_store._connection is sync_conn

    async def test_write_and_filter_with_embeddings_async(self, document_store: IBMDb2DocumentStore) -> None:
        """Round-trip embeddings through the async path."""
        embedding = [0.1] * 768
        docs = [Document(content="test", embedding=embedding)]
        await document_store.write_documents_async(docs)
        retrieved = await document_store.filter_documents_async()
        assert len(retrieved) == 1
        assert retrieved[0].embedding == pytest.approx(embedding, abs=1e-5)

    async def test_delete_all_with_recreate_index_async(self, document_store: IBMDb2DocumentStore) -> None:
        """delete_all_documents_async(recreate_index=True) recreates the table."""
        docs = [Document(content="a"), Document(content="b")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        deleted = await document_store.delete_all_documents_async(recreate_index=True)
        assert deleted == 2
        assert await document_store.count_documents_async() == 0

        # Table must still be usable
        await document_store.write_documents_async([Document(content="c")])
        assert await document_store.count_documents_async() == 1


# ---------------------------------------------------------------------------
# Unit tests — no live DB2 required, use mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_async_store() -> IBMDb2DocumentStore:
    """
    Return an IBMDb2DocumentStore whose async internals are fully mocked out.
    The store is never connected to a real DB2 instance.
    """
    store = IBMDb2DocumentStore(
        database="testdb",
        hostname="localhost",
        username=Secret.from_token("user"),
        password=Secret.from_token("pass"),
        table_name="haystack_mock",
        embedding_dim=768,
    )
    return store


@pytest.mark.asyncio
async def test_close_async_is_idempotent(mock_async_store: IBMDb2DocumentStore) -> None:
    """close_async() on an already-closed store must not raise."""
    assert mock_async_store._async_connection is None
    await mock_async_store.close_async()  # should not raise
    assert mock_async_store._async_connection is None


@pytest.mark.asyncio
async def test_close_async_is_exception_safe(mock_async_store: IBMDb2DocumentStore) -> None:
    """close_async() suppresses exceptions from conn.close()."""
    mock_conn = MagicMock()
    mock_conn.close = AsyncMock(side_effect=RuntimeError("boom"))
    mock_async_store._async_connection = mock_conn
    mock_async_store._async_table_initialized = True

    await mock_async_store.close_async()

    assert mock_async_store._async_connection is None
    assert mock_async_store._async_table_initialized is False


@pytest.mark.asyncio
async def test_write_documents_async_rejects_non_document_items(mock_async_store: IBMDb2DocumentStore) -> None:
    with pytest.raises(ValueError, match="Expected Document objects"):
        await mock_async_store.write_documents_async([{"not": "a document"}])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_write_documents_async_rejects_non_list(mock_async_store: IBMDb2DocumentStore) -> None:
    with pytest.raises(ValueError, match="Expected a list"):
        await mock_async_store.write_documents_async("not a list")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_count_unique_metadata_by_filter_async_empty_fields(
    mock_async_store: IBMDb2DocumentStore,
) -> None:
    result = await mock_async_store.count_unique_metadata_by_filter_async(metadata_fields=None)
    assert result == {}

    result = await mock_async_store.count_unique_metadata_by_filter_async(metadata_fields=[])
    assert result == {}


@pytest.mark.asyncio
async def test_delete_documents_async_noop_on_empty_list(mock_async_store: IBMDb2DocumentStore) -> None:
    """delete_documents_async([]) must not attempt a DB connection."""
    await mock_async_store.delete_documents_async([])
    assert mock_async_store._async_connection is None


@pytest.mark.asyncio
async def test_delete_by_filter_async_noop_on_no_filters(mock_async_store: IBMDb2DocumentStore) -> None:
    count = await mock_async_store.delete_by_filter_async(filters=None)
    assert count == 0
    assert mock_async_store._async_connection is None


@pytest.mark.asyncio
async def test_update_by_filter_async_raises_on_empty_meta(mock_async_store: IBMDb2DocumentStore) -> None:
    with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
        await mock_async_store.update_by_filter_async(
            filters={"field": "meta.x", "operator": "==", "value": "y"}, meta={}
        )


@pytest.mark.asyncio
async def test_update_by_filter_async_noop_on_no_filters(mock_async_store: IBMDb2DocumentStore) -> None:
    result = await mock_async_store.update_by_filter_async(filters=None, meta={"k": "v"})
    assert result == 0
    assert mock_async_store._async_connection is None


# ---------------------------------------------------------------------------
# Mocked-connection unit tests — cover async execution paths
# ---------------------------------------------------------------------------


def _make_mock_cursor(fetchone_return=None, fetchall_return=None, rowcount=0):
    """Build an AsyncMock cursor that behaves like AsyncCursor."""
    cur = MagicMock()
    cur.execute = AsyncMock()
    cur.executemany = AsyncMock()
    cur.fetchone = AsyncMock(return_value=fetchone_return)
    cur.fetchall = AsyncMock(return_value=fetchall_return or [])
    cur.rowcount = rowcount
    # support async with cursor:
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    return cur


def _make_mock_conn(cursor):
    """Build an AsyncMock connection that returns the given cursor."""
    conn = MagicMock()
    conn.cursor = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    conn.close = AsyncMock()
    return conn


@pytest.fixture
def mock_connected_store(mock_async_store: IBMDb2DocumentStore):
    """
    Store with _get_connection_async patched to return a mock connection.
    The cursor returned has sensible defaults that can be overridden per test.
    """
    cur = _make_mock_cursor(fetchone_return=(0,), fetchall_return=[])
    conn = _make_mock_conn(cur)
    mock_async_store._get_connection_async = AsyncMock(return_value=conn)  # type: ignore[method-assign]
    mock_async_store._async_table_initialized = True
    mock_async_store._mock_conn = conn
    mock_async_store._mock_cur = cur
    return mock_async_store


@pytest.mark.asyncio
async def test_count_documents_async_calls_execute(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchone = AsyncMock(return_value=(7,))
    result = await mock_connected_store.count_documents_async()
    assert result == 7
    mock_connected_store._mock_cur.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_documents_by_filter_async_with_filters(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchone = AsyncMock(return_value=(3,))
    result = await mock_connected_store.count_documents_by_filter_async(
        filters={"field": "meta.category", "operator": "==", "value": "A"}
    )
    assert result == 3


@pytest.mark.asyncio
async def test_filter_documents_async_returns_documents(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchall = AsyncMock(
        return_value=[("id1", "content1", json.dumps({"k": "v"}), None)]
    )
    docs = await mock_connected_store.filter_documents_async()
    assert len(docs) == 1
    assert docs[0].id == "id1"
    assert docs[0].content == "content1"
    assert docs[0].meta == {"k": "v"}


@pytest.mark.asyncio
async def test_write_documents_async_insert_path(mock_connected_store: IBMDb2DocumentStore) -> None:
    docs = [Document(content="doc1"), Document(content="doc2")]
    result = await mock_connected_store.write_documents_async(docs)
    assert result == 2
    mock_connected_store._mock_cur.executemany.assert_awaited_once()
    mock_connected_store._mock_conn.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_write_documents_async_skip_policy(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.rowcount = 1
    docs = [Document(content="doc1")]
    result = await mock_connected_store.write_documents_async(docs, policy=DuplicatePolicy.SKIP)
    assert result == 1


@pytest.mark.asyncio
async def test_write_documents_async_overwrite_policy(mock_connected_store: IBMDb2DocumentStore) -> None:
    docs = [Document(content="doc1")]
    result = await mock_connected_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)
    assert result == 1


@pytest.mark.asyncio
async def test_delete_documents_async_executes_delete(mock_connected_store: IBMDb2DocumentStore) -> None:
    await mock_connected_store.delete_documents_async(["id1", "id2"])
    mock_connected_store._mock_cur.execute.assert_awaited_once()
    sql_call = mock_connected_store._mock_cur.execute.call_args[0][0]
    assert "DELETE" in sql_call


@pytest.mark.asyncio
async def test_delete_by_filter_async_returns_rowcount(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.rowcount = 5
    result = await mock_connected_store.delete_by_filter_async(
        filters={"field": "meta.category", "operator": "==", "value": "A"}
    )
    assert result == 5


@pytest.mark.asyncio
async def test_delete_all_documents_async_no_recreate(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchone = AsyncMock(return_value=(4,))
    result = await mock_connected_store.delete_all_documents_async(recreate_index=False)
    assert result == 4
    # Second execute call should be the DELETE
    calls = mock_connected_store._mock_cur.execute.call_args_list
    assert any("DELETE" in str(c) for c in calls)


@pytest.mark.asyncio
async def test_update_by_filter_async_updates_rows(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchall = AsyncMock(return_value=[("id1", json.dumps({"old": "val"}))])
    result = await mock_connected_store.update_by_filter_async(
        filters={"field": "meta.old", "operator": "==", "value": "val"},
        meta={"new_key": "new_val"},
    )
    assert result == 1
    # UPDATE execute should have been called
    calls = mock_connected_store._mock_cur.execute.call_args_list
    assert any("UPDATE" in str(c) for c in calls)


@pytest.mark.asyncio
async def test_get_metadata_fields_info_async_empty(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchall = AsyncMock(return_value=[])
    result = await mock_connected_store.get_metadata_fields_info_async()
    assert result == {}


@pytest.mark.asyncio
async def test_get_metadata_field_min_max_async_returns_none_on_empty(
    mock_connected_store: IBMDb2DocumentStore,
) -> None:
    mock_connected_store._mock_cur.fetchone = AsyncMock(return_value=(None, None))
    result = await mock_connected_store.get_metadata_field_min_max_async("priority")
    assert result == {"min": None, "max": None}


@pytest.mark.asyncio
async def test_get_metadata_field_unique_values_async_empty(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchone = AsyncMock(return_value=(0,))
    mock_connected_store._mock_cur.fetchall = AsyncMock(return_value=[])
    values, count = await mock_connected_store.get_metadata_field_unique_values_async("category")
    assert values == []
    assert count == 0


@pytest.mark.asyncio
async def test_count_unique_metadata_by_filter_async_executes_query(
    mock_connected_store: IBMDb2DocumentStore,
) -> None:
    mock_connected_store._mock_cur.fetchall = AsyncMock(return_value=[("A",), ("B",), ("A",)])
    result = await mock_connected_store.count_unique_metadata_by_filter_async(metadata_fields=["category"])
    assert result == {"category": 2}  # A and B are distinct


@pytest.mark.asyncio
async def test_embedding_retrieval_async_returns_documents(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.fetchall = AsyncMock(return_value=[("id1", "content1", json.dumps({}), None, 0.1)])
    docs = await mock_connected_store._embedding_retrieval_async(query_embedding=[0.1] * 768, top_k=1)
    assert len(docs) == 1
    assert docs[0].id == "id1"
    assert docs[0].score == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_embedding_retrieval_async_handles_zero_vector_error(
    mock_connected_store: IBMDb2DocumentStore,
) -> None:
    mock_connected_store._mock_cur.fetchall = AsyncMock(side_effect=Exception("SQL0801N Division by zero"))
    docs = await mock_connected_store._embedding_retrieval_async(query_embedding=[0.1] * 768, top_k=1)
    assert docs == []


@pytest.mark.asyncio
async def test_transaction_async_rolls_back_on_error(mock_connected_store: IBMDb2DocumentStore) -> None:
    mock_connected_store._mock_cur.execute = AsyncMock(side_effect=Exception("db error"))
    with pytest.raises(DocumentStoreError, match="db error"):
        async with mock_connected_store._transaction_async("test op") as cur:
            await cur.execute("SELECT 1")
    mock_connected_store._mock_conn.rollback.assert_awaited_once()
    mock_connected_store._mock_conn.commit.assert_not_called()
