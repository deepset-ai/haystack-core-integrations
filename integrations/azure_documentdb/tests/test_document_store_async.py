# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)
from pymongo import InsertOne, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError

from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore


async def test_async_crud_methods(mocked_store_collection_async):
    store, collection, _, _ = mocked_store_collection_async
    collection.count_documents = AsyncMock(return_value=2)
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[{"id": "one", "content": "text", "meta": {}}])
    collection.find.return_value = cursor
    collection.delete_many = AsyncMock()

    assert await store.count_documents_async() == 2
    assert await store.filter_documents_async() == [Document(id="one", content="text")]
    await store.delete_documents_async(["one"])

    collection.delete_many.assert_awaited_once_with({"id": {"$in": ["one"]}})


async def test_async_filter_mutation_methods(mocked_store_collection_async):
    store, collection, _, _ = mocked_store_collection_async
    collection.delete_many = AsyncMock()
    collection.delete_many.return_value.deleted_count = 2
    collection.update_many = AsyncMock()
    collection.update_many.return_value.modified_count = 3

    filters = {"field": "meta.kind", "operator": "==", "value": "guide"}
    assert await store.delete_by_filter_async(filters) == 2
    assert await store.update_by_filter_async(filters, {"reviewed": True}) == 3
    await store.delete_all_documents_async()

    collection.delete_many.assert_any_await({"meta.kind": {"$eq": "guide"}})
    collection.update_many.assert_awaited_once_with({"meta.kind": {"$eq": "guide"}}, {"$set": {"meta.reviewed": True}})
    collection.delete_many.assert_any_await({})


@pytest.mark.parametrize(
    ("policy", "operation_type"),
    [
        (DuplicatePolicy.FAIL, InsertOne),
        (DuplicatePolicy.OVERWRITE, ReplaceOne),
        (DuplicatePolicy.SKIP, UpdateOne),
    ],
)
async def test_write_documents_async_policies(mocked_store_collection_async, policy, operation_type):
    store, collection, _, _ = mocked_store_collection_async
    collection.count_documents = AsyncMock(return_value=0)
    collection.bulk_write = AsyncMock()

    assert await store.write_documents_async([Document(id="one", content="text")], policy=policy) == 1
    operation = collection.bulk_write.await_args.args[0][0]
    assert isinstance(operation, operation_type)


async def test_write_documents_async_translates_bulk_write_error(mocked_store_collection_async):
    store, collection, _, _ = mocked_store_collection_async
    collection.bulk_write = AsyncMock(side_effect=BulkWriteError({"writeErrors": [{"code": 11000}]}))
    with pytest.raises(DuplicateDocumentError, match="Duplicate documents found"):
        await store.write_documents_async([Document(id="one")])


async def test_async_retrieval_and_index_creation(mocked_store_collection_async):
    store, collection, _, database = mocked_store_collection_async
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[{"document": {"id": "one", "content": "text", "meta": {}}, "score": 0.8}])
    collection.aggregate = AsyncMock(return_value=cursor)
    database.command = AsyncMock()

    documents = await store._embedding_retrieval_async([0.1], top_k=1)
    assert documents == [Document(id="one", content="text", score=0.8)]
    await store.create_vector_index_async(dimensions=3, kind="vector-ivf", numLists=1)
    assert database.command.await_args.args[0]["indexes"][0]["cosmosSearchOptions"]["dimensions"] == 3


async def test_full_text_retrieval_async(mocked_store_collection_async):
    store, collection, _, _ = mocked_store_collection_async
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[{"id": "one", "content": "Azure", "meta": {}, "score": 1.5}])
    collection.aggregate = AsyncMock(return_value=cursor)

    assert await store._full_text_retrieval_async("azure") == [Document(id="one", content="Azure", score=1.5)]


async def test_async_errors_and_close(mocked_store_collection_async):
    store, collection, client, _ = mocked_store_collection_async
    collection.aggregate = AsyncMock(side_effect=RuntimeError("query failed"))
    with pytest.raises(DocumentStoreError, match="Vector retrieval"):
        await store._embedding_retrieval_async([0.1])
    with pytest.raises(DocumentStoreError, match="Full-text retrieval"):
        await store._full_text_retrieval_async("azure")

    await store.close_async()
    client.close.assert_awaited_once_with()


async def test_async_connection_failure(mocked_store_collection_async):
    store, _, client, _ = mocked_store_collection_async
    client.admin.command.side_effect = RuntimeError("unreachable")
    with pytest.raises(DocumentStoreError, match="Connection to Azure DocumentDB failed"):
        await store.count_documents_async()


@pytest.mark.skipif(
    not os.environ.get("AZURE_DOCUMENTDB_CONNECTION_STRING"), reason="No OSS DocumentDB connection string provided"
)
@pytest.mark.integration
class TestDocumentStoreAsync(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
):
    @pytest_asyncio.fixture
    async def document_store(self, real_collection):
        database_name, collection_name, _ = real_collection
        store = AzureDocumentDBDocumentStore(database_name=database_name, collection_name=collection_name)
        try:
            yield store
        finally:
            await store.close_async()

    async def test_count_not_empty_async(self, document_store: AzureDocumentDBDocumentStore):
        await document_store.write_documents_async([Document(content="one"), Document(content="two")])
        assert await document_store.count_documents_async() == 2

    async def test_write_documents_async(self, document_store: AzureDocumentDBDocumentStore):
        documents = [Document(content="some text")]
        assert await document_store.write_documents_async(documents) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(documents, DuplicatePolicy.FAIL)
