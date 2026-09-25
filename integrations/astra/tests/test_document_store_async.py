# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import operator
import os
import warnings
from unittest import mock

import pytest
from astrapy import AsyncCollection, AsyncDatabase, DataAPIClient
from astrapy.info import CollectionDefinition, CollectionDescriptor
from haystack import Document, Pipeline
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError, MissingDocumentError
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

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore
from haystack_integrations.document_stores.astra.errors import AstraDocumentStoreFilterError


@pytest.fixture
def native_async_mocks(mock_auth):  # noqa: ARG001
    with mock.patch(
        "haystack_integrations.document_stores.astra.astra_client.DataAPIClient", autospec=DataAPIClient
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
        yield store, client, database, collection


@pytest.fixture
async def native_async_store(native_async_mocks):
    try:
        yield native_async_mocks
    finally:
        await native_async_mocks[0].close_async()


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
    assert store._collection is None
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


def test_search_async_across_event_loops(native_async_mocks):
    # e.g. repeated `asyncio.run(pipeline.run_async(...))`: the cached collection's HTTP client and the lock
    # are bound to the loop that first used them, so a new loop must get fresh ones
    store, client, database, _ = native_async_mocks

    async def list_collections():
        await asyncio.sleep(0)  # yield so that concurrent searches contend for the lock
        return []

    database.list_collections.side_effect = list_collections
    collections = [mock.MagicMock(spec=AsyncCollection), mock.MagicMock(spec=AsyncCollection)]
    database.create_collection.side_effect = collections

    async def concurrent_searches():
        await asyncio.gather(store.search_async([0.1] * 4, 2), store.search_async([0.1] * 4, 2))
        return store._async_collection

    assert asyncio.run(concurrent_searches()) is collections[0]
    assert asyncio.run(concurrent_searches()) is collections[1]
    assert client.call_count == 2


async def test_pipeline_close_async_releases_collection(native_async_store):
    store, _, _, collection = native_async_store
    pipeline = Pipeline()
    pipeline.add_component("retriever", AstraEmbeddingRetriever(store))
    await pipeline.run_async({"retriever": {"query_embedding": [0.1] * 4}})
    await pipeline.close_async()
    collection.__aexit__.assert_awaited_once()
    assert store._async_collection is None


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


@pytest.fixture
def mocked_async_store(mock_auth):  # noqa: ARG001
    """Returns (store, collection) with the astrapy async collection mocked out."""
    collection = mock.MagicMock(spec=AsyncCollection)
    with mock.patch.object(AstraDocumentStore, "_get_async_collection", return_value=collection):
        yield AstraDocumentStore(), collection


async def test_write_documents_async_fail_policy_raises_on_duplicate(mocked_async_store):
    store, collection = mocked_async_store
    collection.find_one.return_value = {"_id": "1"}
    with pytest.raises(DuplicateDocumentError, match="already exists"):
        await store.write_documents_async([Document(id="1", content="a")], policy=DuplicatePolicy.FAIL)
    collection.find_one.assert_awaited_once_with({"_id": "1"}, projection={"_id": True})


async def test_write_documents_async_inserts_new_documents(mocked_async_store):
    store, collection = mocked_async_store
    collection.find_one.return_value = None
    collection.insert_many.return_value.inserted_ids = ["1"]
    assert await store.write_documents_async([Document(id="1", content="a", embedding=[0.1] * 4)]) == 1
    inserted = collection.insert_many.call_args.kwargs["documents"][0]
    assert (inserted["_id"], inserted["$vector"]) == ("1", [0.1] * 4)


@pytest.mark.parametrize("updated,expected_count", [({"_id": "1"}, 1), (None, 0)])
async def test_write_documents_async_overwrite_updates_existing(mocked_async_store, updated, expected_count):
    store, collection = mocked_async_store
    collection.find_one.return_value = {"_id": "1"}
    collection.find_one_and_update.return_value = updated
    doc = {"_id": "1", "content": "new"}
    assert await store.write_documents_async([doc], policy=DuplicatePolicy.OVERWRITE) == expected_count
    collection.find_one_and_update.assert_awaited_once_with(
        {"_id": "1"}, {"$set": {"content": "new"}}, projection={"_id": True}
    )
    collection.insert_many.assert_not_awaited()


async def test_count_documents_async(mocked_async_store):
    store, collection = mocked_async_store
    collection.count_documents.return_value = 7
    assert await store.count_documents_async() == 7
    collection.count_documents.assert_awaited_once_with({}, upper_bound=10_000)


async def test_count_documents_by_filter_async(mocked_async_store):
    store, collection = mocked_async_store
    collection.count_documents.return_value = 2
    assert await store.count_documents_by_filter_async({"field": "meta.k", "operator": "==", "value": "v"}) == 2
    collection.count_documents.assert_awaited_once_with({"meta.k": {"$eq": "v"}}, upper_bound=1_000_000_000)


@pytest.mark.parametrize(
    "filters,expected_filter",
    [(None, None), ({"field": "meta.k", "operator": "==", "value": "v"}, {"meta.k": {"$eq": "v"}})],
)
async def test_filter_documents_async(mocked_async_store, filters, expected_filter):
    store, collection = mocked_async_store
    collection.find.return_value.__aiter__.return_value = [{"_id": "1", "content": "a", "meta": {"k": "v"}}]
    assert await store.filter_documents_async(filters) == [Document(id="1", content="a", meta={"k": "v"})]
    collection.find.assert_called_once_with(filter=expected_filter, limit=1000, projection={"*": 1})


async def test_filter_documents_async_invalid_filters(mocked_async_store):
    store, _ = mocked_async_store
    with pytest.raises(AstraDocumentStoreFilterError, match="Filters must be a dictionary"):
        await store.filter_documents_async("bad")  # type: ignore[arg-type]


async def test_get_documents_by_id_async_batches_ids(mocked_async_store):
    store, collection = mocked_async_store
    collection.find.return_value.__aiter__.return_value = [{"_id": "1", "content": "a"}]
    await store.get_documents_by_id_async([str(i) for i in range(21)])
    assert [c.kwargs["filter"] for c in collection.find.call_args_list] == [
        {"_id": {"$in": [str(i) for i in range(20)]}},
        {"_id": {"$in": ["20"]}},
    ]


async def test_get_document_by_id_async_missing_raises(mocked_async_store):
    store, collection = mocked_async_store
    collection.find.return_value.__aiter__.return_value = []
    with pytest.raises(MissingDocumentError, match="does not exist"):
        await store.get_document_by_id_async("missing")


@pytest.mark.parametrize("deleted_count,raises", [(1, False), (0, True)])
async def test_delete_documents_async(mocked_async_store, deleted_count, raises):
    store, collection = mocked_async_store
    collection.find_one.return_value = {"_id": "x"}
    collection.delete_many.return_value.deleted_count = deleted_count
    if raises:
        with pytest.raises(MissingDocumentError, match="does not exist"):
            await store.delete_documents_async(["1"])
    else:
        await store.delete_documents_async(["1"])
    collection.delete_many.assert_awaited_once_with({"_id": {"$in": ["1"]}})


async def test_delete_documents_async_empty_store_is_noop(mocked_async_store):
    store, collection = mocked_async_store
    collection.find_one.return_value = None
    await store.delete_documents_async(["1"])
    collection.delete_many.assert_not_awaited()


async def test_delete_all_documents_async_wraps_exception(mocked_async_store):
    store, collection = mocked_async_store
    collection.delete_many.side_effect = RuntimeError("boom")
    with pytest.raises(DocumentStoreError, match="Failed to delete all documents"):
        await store.delete_all_documents_async()


async def test_delete_by_filter_async(mocked_async_store):
    store, collection = mocked_async_store
    collection.delete_many.return_value.deleted_count = 3
    assert await store.delete_by_filter_async({"field": "meta.k", "operator": "==", "value": "v"}) == 3
    collection.delete_many.assert_awaited_once_with({"meta.k": {"$eq": "v"}})


async def test_update_by_filter_async(mocked_async_store):
    store, collection = mocked_async_store
    collection.update_many.return_value.update_info = {"nModified": 4}
    count = await store.update_by_filter_async(
        filters={"field": "meta.k", "operator": "==", "value": "v"}, meta={"reviewed": True}
    )
    assert count == 4
    collection.update_many.assert_awaited_once_with({"meta.k": {"$eq": "v"}}, {"$set": {"meta.reviewed": True}})


@pytest.mark.parametrize("filters,meta,match", [("bad", {}, "Filters must be"), ({}, "bad", "Meta must be")])
async def test_update_by_filter_async_validation_errors(mocked_async_store, filters, meta, match):
    store, _ = mocked_async_store
    with pytest.raises(AstraDocumentStoreFilterError, match=match):
        await store.update_by_filter_async(filters=filters, meta=meta)


async def test_metadata_methods_async(mocked_async_store):
    store, collection = mocked_async_store
    collection.distinct.return_value = [3, 1, 2, 2]
    collection.find.return_value.__aiter__.return_value = [{"content": "a", "meta": {"priority": 1}}]

    filters = {"field": "meta.k", "operator": "==", "value": "v"}
    assert await store.count_unique_metadata_by_filter_async(filters, ["priority"]) == {"priority": 3}
    assert await store.get_metadata_field_min_max_async("meta.priority") == {"min": 1, "max": 3}
    assert await store.get_metadata_field_unique_values_async("priority", from_=1, size=1) == ([2], 3)
    assert await store.get_metadata_fields_info_async() == {"content": {"type": "text"}, "priority": {"type": "long"}}
    assert collection.distinct.await_args_list == [
        mock.call("meta.priority", filter={"meta.k": {"$eq": "v"}}),
        mock.call("meta.priority"),
        mock.call("meta.priority", filter=None),
    ]
    collection.find.assert_called_once_with(projection={"content": 1, "meta": 1})


@pytest.fixture(scope="class")
def astra_async_store():
    store = AstraDocumentStore(
        collection_name="haystack_test_document_store_async",
        duplicates_policy=DuplicatePolicy.OVERWRITE,
        embedding_dimension=768,
    )
    try:
        yield store
    finally:
        store._get_collection().drop()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN env var not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_API_ENDPOINT", "") == "", reason="ASTRA_DB_API_ENDPOINT env var not set")
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
    @pytest.fixture
    async def document_store(self, astra_async_store: AstraDocumentStore):
        # Each test runs on its own event loop, so the async collection is released after every test.
        await astra_async_store.delete_all_documents_async()
        try:
            yield astra_async_store
        finally:
            await astra_async_store.close_async()

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        received.sort(key=operator.attrgetter("id"))
        expected.sort(key=operator.attrgetter("id"))
        assert received == expected

    async def test_write_documents_async(self, document_store: AstraDocumentStore):
        # The store is configured with `duplicates_policy=OVERWRITE`, which applies when no policy is passed.
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")
        assert await document_store.write_documents_async([doc1]) == 1
        assert await document_store.write_documents_async([doc2]) == 1
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [doc2])

    async def test_delete_documents_non_existing_document_async(self, document_store: AstraDocumentStore):
        # Override: Astra raises when none of the given ids exist, mirroring the sync `delete_documents`.
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        with pytest.raises(MissingDocumentError):
            await document_store.delete_documents_async(["non_existing_id"])
        assert await document_store.count_documents_async() == 1

    async def test_get_metadata_field_unique_values_distinct_types_async(self, document_store: AstraDocumentStore):
        # Override: one field per type, see `TestDocumentStore.test_get_metadata_field_unique_values_distinct_types`.
        docs = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        await document_store.write_documents_async(docs)
        for field, expected in [
            ("priority_int", 1),
            ("priority_str", "1"),
            ("priority_float", 1.5),
            ("priority_bool", True),
        ]:
            values, count = await document_store.get_metadata_field_unique_values_async(metadata_field=field)
            assert (values, count) == ([expected], 1)
            assert type(values[0]) is type(expected)

    async def test_get_documents_by_id_async(self, document_store: AstraDocumentStore):
        # More than one batch of 20 ids.
        docs = [Document(id=str(i), content=f"doc {i}") for i in range(25)]
        await document_store.write_documents_async(docs)
        assert await document_store.get_document_by_id_async("3") == Document(id="3", content="doc 3")
        self.assert_documents_are_equal(await document_store.get_documents_by_id_async([d.id for d in docs]), docs)
        with pytest.raises(MissingDocumentError):
            await document_store.get_document_by_id_async("missing")
