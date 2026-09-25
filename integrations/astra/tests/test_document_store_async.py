# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from unittest import mock

import pytest
from astrapy import AsyncCollection, AsyncDatabase, DataAPIClient
from astrapy.info import CollectionDefinition, CollectionDescriptor
from haystack import Document

from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.fixture
async def native_async_store(mock_auth):  # noqa: ARG001
    with mock.patch(
        "haystack_integrations.document_stores.astra.document_store.DataAPIClient", autospec=DataAPIClient
    ) as client:
        database = mock.MagicMock(spec=AsyncDatabase)
        collection = mock.MagicMock(spec=AsyncCollection)
        database.__aenter__.return_value = database
        database.list_collections.return_value = []
        database.create_collection.return_value = collection
        database.get_collection.return_value = collection
        client.return_value.get_async_database.return_value = database
        store = AstraDocumentStore(
            collection_name="custom", embedding_dimension=4, similarity="dot_product", namespace="keyspace"
        )
        try:
            yield store, client, database, collection
        finally:
            await store.close_async()


@pytest.mark.parametrize("existing", [False, True])
async def test_async_configuration(native_async_store, existing):
    store, client, database, collection = native_async_store
    client.assert_not_called()
    if existing:
        database.list_collections.return_value = [
            CollectionDescriptor(
                name="custom",
                definition=CollectionDefinition(indexing={"deny": ["metadata._node_content", "content"]}),
                raw_descriptor={},
            )
        ]
    assert await store._get_async_collection() is collection
    assert store._index is None
    serdes = client.call_args.kwargs["api_options"].serdes_options
    assert serdes.binary_encode_vectors is False
    assert serdes.custom_datatypes_in_reading is False
    client.return_value.get_database.assert_not_called()
    client.return_value.get_async_database.assert_called_once_with(
        api_endpoint="http://example.com", token="test_token", keyspace="keyspace"
    )
    database.list_collections.assert_awaited_once()
    if existing:
        database.create_collection.assert_not_awaited()
        database.get_collection.assert_called_once_with("custom")
    else:
        database.create_collection.assert_awaited_once_with(
            name="custom",
            definition={
                "vector": {"dimension": 4, "metric": "dot_product"},
                "indexing": {"deny": ["metadata._node_content", "content"]},
            },
        )
    collection.__aexit__.assert_not_awaited()
    database.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("filters", [None, {"field": "meta.category", "operator": "==", "value": "news"}])
async def test_search_async_uses_native_api(native_async_store, filters):
    store, client, _, collection = native_async_store
    collection.find.return_value.__aiter__.return_value = [
        {"_id": "1", "content": "text", "$vector": [0.1] * 4, "meta": {"category": "news"}, "$similarity": 0.9}
    ]
    result = await store.search_async([0.2] * 4, 2, filters)
    assert result == [Document(id="1", content="text", embedding=[0.1] * 4, meta={"category": "news"}, score=0.9)]
    client.return_value.get_database.assert_not_called()
    collection.find.assert_called_once_with(
        filter={"meta.category": {"$eq": "news"}} if filters else None,
        sort={"$vector": [0.2] * 4},
        limit=2,
        include_similarity=True,
        projection={"*": 1},
    )


async def test_search_async_reuses_collection_and_reopens_after_close(native_async_store):
    store, client, database, collection = native_async_store
    response = {"_id": "1", "content": "text", "$vector": [0.1] * 4, "meta": {"category": "news"}}
    collection.find.return_value.__aiter__.return_value = [response]
    expected = [Document(id="1", content="text", embedding=[0.1] * 4, meta={"category": "news"})]
    assert await store.search_async([0.1] * 4, 2) == expected
    assert await store.search_async([0.1] * 4, 2) == expected
    assert client.call_count == 1
    database.list_collections.assert_awaited_once()
    collection.__aexit__.assert_not_awaited()

    await store.close_async()
    await store.close_async()
    collection.__aexit__.assert_awaited_once()
    assert store._async_collection is None

    reopened = mock.MagicMock(spec=AsyncCollection)
    reopened.find.return_value.__aiter__.return_value = [response]
    database.create_collection.return_value = reopened
    assert await store.search_async([0.1] * 4, 2) == expected
    assert store._async_collection is reopened
    assert database.list_collections.await_count == 2


async def test_concurrent_searches_initialize_once(native_async_store):
    store, _, database, collection = native_async_store

    async def list_collections():
        await asyncio.sleep(0)
        return []

    database.list_collections.side_effect = list_collections
    assert await asyncio.gather(store.search_async([0.1] * 4, 2), store.search_async([0.1] * 4, 2)) == [[], []]
    database.list_collections.assert_awaited_once()
    database.create_collection.assert_awaited_once()
    assert collection.find.call_count == 2


async def test_close_async_before_search_does_not_initialize(native_async_store):
    store, client, _, _ = native_async_store
    await store.close_async()
    client.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("query failed"), asyncio.CancelledError()])
async def test_close_async_after_search_failure(native_async_store, failure):
    store, _, _, collection = native_async_store

    async def failing_cursor():
        yield {"_id": "1", "content": "first page"}
        raise failure

    collection.find.return_value = failing_cursor()
    with pytest.raises(type(failure)):
        await store.search_async([0.1] * 4, 2)
    await store.close_async()
    collection.__aexit__.assert_awaited_once()
    assert store._async_collection is None


@pytest.mark.parametrize("operation", ["list_collections", "create_collection"])
async def test_search_async_releases_database_on_initialization_failure(native_async_store, operation):
    store, _, database, collection = native_async_store
    getattr(database, operation).side_effect = RuntimeError("initialization failed")
    with pytest.raises(RuntimeError, match="initialization failed"):
        await store.search_async([0.1] * 4, 2)
    database.__aexit__.assert_awaited_once()
    assert store._async_collection is None
    collection.find.assert_not_called()


@pytest.mark.parametrize(
    "indexing,warning_match",
    [(None, "having indexing turned on"), ({"deny": ["other"]}, "unexpected 'indexing' settings")],
)
async def test_existing_collection_warns_only_on_initialization(native_async_store, indexing, warning_match):
    store, _, database, _ = native_async_store
    database.list_collections.return_value = [
        CollectionDescriptor(name="custom", definition=CollectionDefinition(indexing=indexing), raw_descriptor={})
    ]
    with pytest.warns(UserWarning, match=warning_match) as recorded:
        await store.search_async([0.1] * 4, 2)
    assert recorded[0].filename == __file__
    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter("always")
        await store.search_async([0.1] * 4, 2)
    assert not repeated
    database.list_collections.assert_awaited_once()


async def test_search_async_empty_results(native_async_store, caplog):
    store, _, _, collection = native_async_store
    collection.find.return_value.__aiter__.return_value = []
    assert await store.search_async([0.1] * 4, 2) == []
    assert "No documents found" in caplog.text
