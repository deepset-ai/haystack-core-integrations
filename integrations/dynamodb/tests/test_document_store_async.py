# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
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

from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

from .conftest import (
    EMBEDDING_DIMENSION,
    assert_documents_equal_ignoring_order,
    client_error,
    make_store,
    table_description,
)


async def _pages(*pages: dict) -> AsyncIterator[dict]:
    for page in pages:
        yield page


def _async_client(*, pages: list[dict] | None = None) -> AsyncMock:
    """
    Mocks an aiobotocore client: API calls are awaited, while `get_paginator` and `get_waiter`
    return synchronously and the paginator yields `pages` asynchronously.
    """
    client = AsyncMock()
    client.get_paginator = MagicMock()
    client.get_paginator.return_value.paginate.return_value = _pages(*(pages or []))
    client.get_waiter = MagicMock()
    client.get_waiter.return_value.wait = AsyncMock()
    return client


def _patch_async_client(store: DynamoDBDocumentStore, client: AsyncMock) -> Any:
    @asynccontextmanager
    async def fake_client() -> AsyncIterator[AsyncMock]:
        yield client

    return patch.object(store, "_async_client", fake_client)


def _ready(store: DynamoDBDocumentStore) -> DynamoDBDocumentStore:
    store._table_ready = True
    return store


class TestDynamoDBDocumentStoreAsync:
    async def test_ensure_table_async_creates_missing_table_and_probes_search(self) -> None:
        store = make_store()
        store.search_available_poll_interval = 0.0
        client = _async_client()
        client.describe_table.side_effect = [
            client_error("ResourceNotFoundException", "DescribeTable"),
            table_description(),
        ]
        client.search_vectors.side_effect = [client_error("ResourceNotFoundException", "SearchVectors"), {}]
        with _patch_async_client(store, client):
            await store._ensure_table_async()
        _, kwargs = client.create_table.call_args
        assert kwargs["VectorIndexes"][0]["IndexName"] == "test_index"
        client.get_waiter.return_value.wait.assert_awaited_once_with(TableName="test_docs")
        assert client.search_vectors.await_count == 2
        assert store._table_ready is True

    async def test_ensure_table_async_validates_existing_table(self) -> None:
        store = make_store()
        client = _async_client()
        client.describe_table.return_value = table_description(dimensions=768)
        with _patch_async_client(store, client), pytest.raises(ValueError, match="Dimensions=768"):
            await store._ensure_table_async()
        client.create_table.assert_not_called()

    async def test_ensure_table_async_raises_when_missing_and_creation_disabled(self) -> None:
        store = make_store(create_table_if_not_exists=False)
        client = _async_client()
        client.describe_table.side_effect = client_error("ResourceNotFoundException", "DescribeTable")
        with _patch_async_client(store, client), pytest.raises(ValueError, match="create_table_if_not_exists"):
            await store._ensure_table_async()

    async def test_ensure_table_async_is_a_noop_once_ready(self) -> None:
        store = _ready(make_store())
        client = _async_client()
        with _patch_async_client(store, client):
            await store._ensure_table_async()
        client.describe_table.assert_not_called()

    async def test_count_documents_async(self) -> None:
        store = _ready(make_store())
        client = _async_client(pages=[{"Count": 2}, {"Count": 3}])
        with _patch_async_client(store, client):
            assert await store.count_documents_async() == 5
        _, kwargs = client.get_paginator.return_value.paginate.call_args
        assert kwargs["Select"] == "COUNT"
        assert kwargs["ConsistentRead"] is True

    async def test_filter_documents_async_applies_metadata_filter(self) -> None:
        store = _ready(make_store())
        client = _async_client(
            pages=[
                {
                    "Items": [
                        {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}},
                        {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
                    ]
                }
            ]
        )
        with _patch_async_client(store, client):
            docs = await store.filter_documents_async({"field": "meta.topic", "operator": "==", "value": "db"})
        assert [d.id for d in docs] == ["2"]

    async def test_filter_documents_async_without_filters_returns_all(self) -> None:
        store = _ready(make_store())
        client = _async_client(pages=[{"Items": [{"id": {"S": "1"}, "payload": {"S": '{"content": "a"}'}}]}])
        with _patch_async_client(store, client):
            docs = await store.filter_documents_async()
        assert [d.content for d in docs] == ["a"]

    @pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.NONE])
    async def test_write_documents_async_fail_policy_raises_on_duplicate(self, policy: DuplicatePolicy) -> None:
        store = _ready(make_store())
        client = _async_client()
        client.put_item.side_effect = client_error("ConditionalCheckFailedException", "PutItem")
        with _patch_async_client(store, client), pytest.raises(DuplicateDocumentError):
            await store.write_documents_async([Document(id="1", content="hello")], policy=policy)
        _, kwargs = client.put_item.call_args
        assert kwargs["ConditionExpression"] == "attribute_not_exists(#id)"

    async def test_write_documents_async_skip_policy_counts_only_new_documents(self) -> None:
        store = _ready(make_store())
        client = _async_client()
        client.put_item.side_effect = [client_error("ConditionalCheckFailedException", "PutItem"), {}]
        with _patch_async_client(store, client):
            written = await store.write_documents_async(
                [Document(id="1", content="dup"), Document(id="2", content="new")], policy=DuplicatePolicy.SKIP
            )
        assert written == 1

    async def test_write_documents_async_overwrite_policy_writes_unconditionally(self) -> None:
        store = _ready(make_store())
        client = _async_client()
        with _patch_async_client(store, client):
            written = await store.write_documents_async(
                [Document(id="1", content="hello", embedding=[0.1, 0.2, 0.3])], policy=DuplicatePolicy.OVERWRITE
            )
        assert written == 1
        _, kwargs = client.put_item.call_args
        assert "ConditionExpression" not in kwargs
        assert kwargs["Item"]["embedding"] == {"L": [{"N": "0.1"}, {"N": "0.2"}, {"N": "0.3"}]}

    async def test_write_documents_async_rejects_invalid_input(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            await store.write_documents_async(["not a document"])  # type: ignore[list-item]
        assert await store.write_documents_async([]) == 0

    async def test_delete_documents_async_deletes_each_id(self) -> None:
        store = _ready(make_store())
        client = _async_client()
        with _patch_async_client(store, client):
            await store.delete_documents_async(["1", "2"])
        assert [c.kwargs["Key"] for c in client.delete_item.call_args_list] == [{"id": {"S": "1"}}, {"id": {"S": "2"}}]

    async def test_delete_all_documents_async(self) -> None:
        store = _ready(make_store())
        client = _async_client(pages=[{"Items": [{"id": {"S": "1"}}]}, {"Items": [{"id": {"S": "2"}}]}])
        with _patch_async_client(store, client):
            await store.delete_all_documents_async()
        _, kwargs = client.get_paginator.return_value.paginate.call_args
        assert kwargs["ProjectionExpression"] == "#id"
        assert client.delete_item.await_count == 2

    async def test_delete_by_filter_async(self) -> None:
        store = _ready(make_store())
        client = _async_client(
            pages=[
                {
                    "Items": [
                        {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}},
                        {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
                    ]
                }
            ]
        )
        with _patch_async_client(store, client):
            deleted = await store.delete_by_filter_async({"field": "meta.topic", "operator": "==", "value": "ai"})
        assert deleted == 1
        assert client.delete_item.call_args.kwargs["Key"] == {"id": {"S": "1"}}

    async def test_delete_by_filter_async_rejects_empty_filters(self) -> None:
        with pytest.raises(ValueError, match="use delete_all_documents_async"):
            await make_store().delete_by_filter_async({})

    async def test_update_by_filter_async_merges_meta(self) -> None:
        store = _ready(make_store())
        client = _async_client(
            pages=[{"Items": [{"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}}]}]
        )
        with _patch_async_client(store, client):
            updated = await store.update_by_filter_async(
                {"field": "meta.topic", "operator": "==", "value": "ai"}, meta={"reviewed": True}
            )
        assert updated == 1
        written = json.loads(client.put_item.call_args.kwargs["Item"]["payload"]["S"])
        assert written["meta"] == {"topic": "ai", "reviewed": True}

    async def test_embedding_retrieval_async_returns_scored_documents_in_order(self) -> None:
        store = _ready(make_store())
        client = _async_client()
        client.search_vectors.return_value = {
            "SearchResults": [
                {"Item": {"id": {"S": "same"}, "payload": {"S": '{"content": "a"}'}}, "Score": 0.0},
                {"Item": {"id": {"S": "orth"}, "payload": {"S": '{"content": "b"}'}}, "Score": 1.0},
            ]
        }
        with _patch_async_client(store, client):
            docs = await store._embedding_retrieval_async(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        assert [d.id for d in docs] == ["same", "orth"]
        assert [d.score for d in docs] == [pytest.approx(1.0), pytest.approx(0.5)]
        _, kwargs = client.search_vectors.call_args
        assert kwargs["SearchVector"] == [{"N": "1.0"}, {"N": "0.0"}, {"N": "0.0"}]
        assert kwargs["TopK"] == 2

    async def test_embedding_retrieval_async_applies_filters_within_the_dynamodb_limit(self) -> None:
        store = _ready(make_store())
        client = _async_client()
        client.search_vectors.return_value = {
            "SearchResults": [
                {"Item": {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"k": 1}}'}}, "Score": 0.1},
                {"Item": {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"k": 2}}'}}, "Score": 0.2},
            ]
        }
        with _patch_async_client(store, client):
            docs = await store._embedding_retrieval_async(
                query_embedding=[1.0, 0.0, 0.0], top_k=5, filters={"field": "meta.k", "operator": "==", "value": 2}
            )
        assert [d.id for d in docs] == ["2"]
        assert client.search_vectors.call_args.kwargs["TopK"] == 100

    async def test_embedding_retrieval_async_validates_input(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="top_k must be between 1 and 100"):
            await store._embedding_retrieval_async(query_embedding=[1.0, 0.0, 0.0], top_k=101)
        with pytest.raises(ValueError, match="has 2 dimensions"):
            await store._embedding_retrieval_async(query_embedding=[1.0, 0.0])


@pytest.mark.integration
class TestDynamoDBDocumentStoreAsyncIntegration(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    UpdateByFilterAsyncTest,
):
    """
    Runs the async half of the protocol against a real DynamoDB table; see `conftest.py`.
    """

    @pytest.fixture
    def document_store(self, clean_store: DynamoDBDocumentStore) -> DynamoDBDocumentStore:
        return clean_store

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        assert_documents_equal_ignoring_order(received, expected)

    async def test_write_documents_async(self, document_store: DynamoDBDocumentStore) -> None:
        docs = [Document(content="doc1"), Document(content="doc2")]
        assert await document_store.write_documents_async(docs) == 2

    async def test_embedding_retrieval_async_ranks_by_similarity(self, document_store: DynamoDBDocumentStore) -> None:
        near = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
        far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSION - 2)
        await document_store.write_documents_async(
            [Document(id="near", content="near", embedding=near), Document(id="far", content="far", embedding=far)]
        )
        results = await document_store._embedding_retrieval_async(query_embedding=near, top_k=2)
        assert [d.id for d in results] == ["near", "far"]
        assert results[0].score == pytest.approx(1.0, abs=1e-3)
        assert results[1].score == pytest.approx(0.5, abs=1e-3)
