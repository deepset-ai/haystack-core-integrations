# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from haystack.dataclasses import Document
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

from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
from haystack_integrations.document_stores.falkordb import document_store as document_store_module

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_async_falkordb(monkeypatch):
    constructor = MagicMock()
    client = MagicMock()
    graph = MagicMock()
    client.select_graph.return_value = graph
    client.aclose = AsyncMock()
    graph.query = AsyncMock(return_value=MagicMock(result_set=[]))
    constructor.return_value = client
    monkeypatch.setattr(document_store_module, "AsyncFalkorDB", constructor)
    return constructor, client, graph


class TestFalkorDBDocumentStoreAsyncUnit:
    @pytest.mark.asyncio
    async def test_warm_up_async_initializes_client_and_schema(self, mock_async_falkordb) -> None:
        constructor, client, graph = mock_async_falkordb
        store = FalkorDBDocumentStore()

        await store.warm_up_async()

        constructor.assert_called_once()
        assert store.async_client is client
        assert store.async_graph is graph
        assert store.async_initialized is True
        assert graph.query.await_count == 2

    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self, mock_async_falkordb) -> None:
        constructor, _, graph = mock_async_falkordb
        store = FalkorDBDocumentStore()

        await store.warm_up_async()
        await store.warm_up_async()

        constructor.assert_called_once()
        assert graph.query.await_count == 2

    @pytest.mark.asyncio
    async def test_close_then_warm_up_async_reopens(self, mock_async_falkordb) -> None:
        constructor, client, graph = mock_async_falkordb
        store = FalkorDBDocumentStore()
        await store.warm_up_async()

        await store.close_async()
        await store.warm_up_async()

        assert constructor.call_count == 2
        client.aclose.assert_awaited_once()
        assert store.async_client is client
        assert store.async_graph is graph
        assert store.async_initialized is True
        assert graph.query.await_count == 4


@pytest.mark.integration
@pytest.mark.asyncio
class TestFalkorDBDocumentStoreAsync(
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
    def assert_documents_are_equal(received: list[Document], expected: list[Document]) -> None:
        """Compare documents while allowing for FalkorDB's float32 embeddings."""
        assert len(received) == len(expected), f"Expected {len(expected)} documents but got {len(received)}"
        received_sorted = sorted(received, key=lambda document: document.id)
        expected_sorted = sorted(expected, key=lambda document: document.id)
        for received_document, expected_document in zip(received_sorted, expected_sorted, strict=True):
            assert received_document.id == expected_document.id
            assert received_document.content == expected_document.content
            assert received_document.meta == expected_document.meta
            assert (received_document.embedding is None) == (expected_document.embedding is None)

    @pytest_asyncio.fixture
    async def document_store(self) -> AsyncGenerator[FalkorDBDocumentStore, None]:
        graph_name = f"test_async_graph_{uuid.uuid4().hex}"
        store = FalkorDBDocumentStore(
            host=os.environ.get("FALKORDB_HOST", "localhost"),
            port=int(os.environ.get("FALKORDB_PORT", "6379")),
            graph_name=graph_name,
            embedding_dim=768,
            recreate_graph=True,
        )
        yield store

        try:
            if store.async_graph is not None:
                await store.async_graph.delete()
        except Exception:
            logger.debug("Could not delete graph %s during teardown", graph_name)
        finally:
            await store.close_async()

    async def test_write_documents_async(self, document_store: FalkorDBDocumentStore) -> None:
        """FalkorDB defaults to failing on duplicates; verify a normal default write."""
        document = Document(content="test doc")
        assert await document_store.write_documents_async([document]) == 1
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [document])

    async def test_get_metadata_field_unique_values_distinct_types_async(
        self, document_store: FalkorDBDocumentStore
    ) -> None:
        """Avoid Cypher DISTINCT collapsing a whole-number float and an equal integer."""
        documents = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        await document_store.write_documents_async(documents)

        int_values, int_count = await document_store.get_metadata_field_unique_values_async("priority_int")
        str_values, str_count = await document_store.get_metadata_field_unique_values_async("priority_str")
        float_values, float_count = await document_store.get_metadata_field_unique_values_async("priority_float")
        bool_values, bool_count = await document_store.get_metadata_field_unique_values_async("priority_bool")

        assert (int_count, str_count, float_count, bool_count) == (1, 1, 1, 1)
        assert int_values == [1] and type(int_values[0]) is int
        assert str_values == ["1"] and type(str_values[0]) is str
        assert float_values == [1.5] and type(float_values[0]) is float
        assert bool_values == [True] and type(bool_values[0]) is bool

    async def test_close_async_and_reopen(self, document_store: FalkorDBDocumentStore) -> None:
        assert await document_store.count_documents_async() == 0
        await document_store.close_async()
        assert document_store.async_client is None
        await document_store.warm_up_async()
        assert document_store.async_client is not None
        assert await document_store.count_documents_async() == 0
