# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseExtendedTests

from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore
from haystack_integrations.document_stores.dynamodb.document_store import SEARCH_VECTORS_MAX_TOP_K

from .conftest import (
    EMBEDDING_DIMENSION,
    assert_documents_equal_ignoring_order,
    client_error,
    make_store,
    table_description,
)


class TestDynamoDBDocumentStore:
    def test_init_default(self) -> None:
        store = DynamoDBDocumentStore()
        assert store.table_name == "haystack_documents"
        assert store.index_name == "haystack_vector_index"
        assert store.embedding_dimension == 768
        assert store.similarity_function == "cosine"

    def test_init_rejects_non_cosine_similarity(self) -> None:
        with pytest.raises(ValueError, match="supports only 'cosine'"):
            DynamoDBDocumentStore(similarity_function="dot_product")

    def test_ensure_table_uses_compatible_existing_table(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.return_value = table_description()
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        assert store._table_ready is True
        mock_client.create_table.assert_not_called()

    def test_ensure_table_rejects_existing_table_without_vector_index(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.return_value = table_description(index_name="some_other_index")
        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ValueError, match="has no vector index named 'test_index'"),
        ):
            store._ensure_table()
        assert store._table_ready is False

    @pytest.mark.parametrize(
        ("overrides", "expected_message"),
        [
            ({"dimensions": 768}, r"Dimensions=768 \(expected 3\)"),
            ({"distance_function": "EUCLIDEAN"}, r"DistanceFunction=EUCLIDEAN \(expected COSINE\)"),
            ({"vector_attribute": "vec"}, r"VectorAttribute='vec' \(expected 'embedding'\)"),
        ],
    )
    def test_ensure_table_rejects_incompatible_vector_index(self, overrides: dict, expected_message: str) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.return_value = table_description(**overrides)
        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ValueError, match=expected_message),
        ):
            store._ensure_table()

    def test_ensure_table_rejects_existing_table_with_sort_key(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.return_value = table_description(
            key_schema=[{"AttributeName": "id", "KeyType": "HASH"}, {"AttributeName": "ts", "KeyType": "RANGE"}]
        )
        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ValueError, match="single partition key named 'id' and no sort key"),
        ):
            store._ensure_table()

    def test_ensure_table_creates_missing_table_with_inline_vector_index(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = [
            client_error("ResourceNotFoundException", "DescribeTable"),
            table_description(),
        ]
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        _, kwargs = mock_client.create_table.call_args
        assert kwargs["TableName"] == "test_docs"
        assert kwargs["KeySchema"] == [{"AttributeName": "id", "KeyType": "HASH"}]
        assert "GlobalSecondaryIndexes" not in kwargs
        assert kwargs["VectorIndexes"] == [
            {
                "IndexName": "test_index",
                "VectorAttribute": {"AttributeName": "embedding"},
                "Dimensions": 3,
                "DistanceFunction": "COSINE",
                "Projection": {"ProjectionType": "ALL"},
            }
        ]
        mock_client.get_waiter.assert_called_once_with("table_exists")
        mock_client.get_waiter.return_value.wait.assert_called_once_with(TableName="test_docs")
        assert store._table_ready is True

    def test_ensure_table_waits_until_search_vectors_accepts_requests(self) -> None:
        """
        Right after the index turns ACTIVE, `SearchVectors` can still answer
        ResourceNotFoundException for a few seconds; a freshly created table must not be
        reported ready before that window has passed.
        """
        store = make_store()
        store.search_available_poll_interval = 0.0
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = [
            client_error("ResourceNotFoundException", "DescribeTable"),
            table_description(),
        ]
        mock_client.search_vectors.side_effect = [
            client_error("ResourceNotFoundException", "SearchVectors"),
            client_error("ResourceNotFoundException", "SearchVectors"),
            {"SearchResults": []},
        ]
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        assert mock_client.search_vectors.call_count == 3
        _, kwargs = mock_client.search_vectors.call_args
        assert kwargs["TopK"] == 1
        assert len(kwargs["SearchVector"]) == 3
        assert store._table_ready is True

    def test_ensure_table_probe_gives_up_on_unexpected_errors(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = [
            client_error("ResourceNotFoundException", "DescribeTable"),
            table_description(),
        ]
        mock_client.search_vectors.side_effect = client_error("ValidationException", "SearchVectors")
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        assert mock_client.search_vectors.call_count == 1
        assert store._table_ready is True

    def test_ensure_table_does_not_probe_pre_existing_tables(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.return_value = table_description()
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        mock_client.search_vectors.assert_not_called()

    def test_ensure_table_tolerates_concurrent_creation(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = [
            client_error("ResourceNotFoundException", "DescribeTable"),
            table_description(),
        ]
        mock_client.create_table.side_effect = client_error("ResourceInUseException", "CreateTable")
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        assert store._table_ready is True

    def test_ensure_table_raises_when_missing_and_creation_disabled(self) -> None:
        store = make_store(create_table_if_not_exists=False)
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = client_error("ResourceNotFoundException", "DescribeTable")
        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(ValueError, match="does not exist and create_table_if_not_exists is False"),
        ):
            store._ensure_table()
        mock_client.create_table.assert_not_called()

    def test_ensure_table_reraises_unexpected_describe_errors(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = client_error("AccessDeniedException", "DescribeTable")
        with patch.object(store, "_get_client", return_value=mock_client), pytest.raises(ClientError):
            store._ensure_table()

    def test_ensure_table_waits_for_backfilling_index(self) -> None:
        store = make_store()
        store.index_ready_poll_interval = 0.0
        mock_client = MagicMock()
        mock_client.describe_table.side_effect = [
            table_description(index_status="CREATING", backfilling=True),
            table_description(index_status="ACTIVE", backfilling=True),
            table_description(index_status="ACTIVE"),
        ]
        with patch.object(store, "_get_client", return_value=mock_client):
            store._ensure_table()
        assert mock_client.describe_table.call_count == 3
        assert store._table_ready is True

    def test_ensure_table_times_out_when_index_never_becomes_ready(self) -> None:
        store = make_store()
        store.index_ready_timeout = 0.0
        mock_client = MagicMock()
        mock_client.describe_table.return_value = table_description(index_status="CREATING")
        with (
            patch.object(store, "_get_client", return_value=mock_client),
            pytest.raises(TimeoutError, match="did not become queryable"),
        ):
            store._ensure_table()

    def test_count_documents(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Count": 5}]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            assert store.count_documents() == 5

    def test_write_documents_rejects_non_document(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            store.write_documents([{"not": "a document"}])  # type: ignore[list-item]

    def test_write_documents_empty_list(self) -> None:
        store = make_store()
        assert store.write_documents([]) == 0

    @pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.NONE])
    def test_write_documents_fail_policy_raises_on_duplicate(self, policy: DuplicatePolicy) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.put_item.side_effect = client_error("ConditionalCheckFailedException", "PutItem")
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            with pytest.raises(DuplicateDocumentError):
                store.write_documents([Document(id="1", content="hello")], policy=policy)
        _, kwargs = mock_client.put_item.call_args
        assert kwargs["ConditionExpression"] == "attribute_not_exists(#id)"
        assert kwargs["ExpressionAttributeNames"] == {"#id": "id"}
        # no separate existence check: the conditional write is the duplicate check
        mock_client.get_item.assert_not_called()

    def test_write_documents_fail_policy_keeps_documents_written_before_the_duplicate(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.put_item.side_effect = [{}, client_error("ConditionalCheckFailedException", "PutItem")]
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            with pytest.raises(DuplicateDocumentError, match="id '2' already exists"):
                store.write_documents([Document(id="1", content="a"), Document(id="2", content="b")])
        assert mock_client.put_item.call_count == 2

    def test_write_documents_skip_policy_skips_duplicate(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.put_item.side_effect = [client_error("ConditionalCheckFailedException", "PutItem"), {}]
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            written = store.write_documents(
                [Document(id="1", content="dup"), Document(id="2", content="new")], policy=DuplicatePolicy.SKIP
            )
        assert written == 1
        assert mock_client.put_item.call_count == 2

    def test_write_documents_overwrite_policy_writes_unconditionally(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            written = store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        _, kwargs = mock_client.put_item.call_args
        assert "ConditionExpression" not in kwargs
        assert kwargs["Item"]["id"] == {"S": "1"}

    def test_write_documents_reraises_unexpected_put_errors(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.put_item.side_effect = client_error("ProvisionedThroughputExceededException", "PutItem")
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            with pytest.raises(ClientError):
                store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)

    def test_delete_all_documents_deletes_every_scanned_id(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Items": [{"id": {"S": "1"}}]}, {"Items": [{"id": {"S": "2"}}]}]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            store.delete_all_documents()
        _, scan_kwargs = mock_paginator.paginate.call_args
        assert scan_kwargs["ProjectionExpression"] == "#id"
        assert scan_kwargs["ConsistentRead"] is True
        assert [c.kwargs["Key"] for c in mock_client.delete_item.call_args_list] == [
            {"id": {"S": "1"}},
            {"id": {"S": "2"}},
        ]

    def test_delete_by_filter_deletes_only_matching_documents(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Items": [
                    {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}},
                    {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
                    {"id": {"S": "3"}, "payload": {"S": '{"content": "c", "meta": {"topic": "ai"}}'}},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            deleted = store.delete_by_filter({"field": "meta.topic", "operator": "==", "value": "ai"})
        assert deleted == 2
        assert [c.kwargs["Key"]["id"]["S"] for c in mock_client.delete_item.call_args_list] == ["1", "3"]

    def test_delete_by_filter_rejects_empty_filters(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="use delete_all_documents"):
            store.delete_by_filter({})

    def test_update_by_filter_merges_meta_into_matching_documents(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Items": [
                    {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai", "year": 2024}}'}},
                    {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            updated = store.update_by_filter(
                {"field": "meta.topic", "operator": "==", "value": "ai"}, meta={"reviewed": True, "year": 2025}
            )
        assert updated == 1
        mock_client.put_item.assert_called_once()
        _, kwargs = mock_client.put_item.call_args
        assert "ConditionExpression" not in kwargs
        written = json.loads(kwargs["Item"]["payload"]["S"])
        assert written["meta"] == {"topic": "ai", "year": 2025, "reviewed": True}

    def test_update_by_filter_rejects_empty_filters(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="filters must not be empty"):
            store.update_by_filter({}, meta={"x": 1})

    def test_delete_documents_empty_list_is_noop(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        with patch.object(store, "_get_client", return_value=mock_client):
            store.delete_documents([])
            mock_client.delete_item.assert_not_called()

    def test_delete_documents_calls_delete_item_per_id(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            store.delete_documents(["1", "2"])
            assert mock_client.delete_item.call_count == 2

    def test_filter_documents_no_filter_returns_all(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        item = {"id": {"S": "1"}, "payload": {"S": '{"content": "hello"}'}}
        mock_paginator.paginate.return_value = [{"Items": [item]}]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store.filter_documents(None)
            assert len(docs) == 1
            assert docs[0].id == "1"
            assert docs[0].content == "hello"

    def test_filter_documents_applies_metadata_filter(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        items = [
            {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}},
            {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
        ]
        mock_paginator.paginate.return_value = [{"Items": items}]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store.filter_documents({"field": "meta.topic", "operator": "==", "value": "ai"})
            assert [d.id for d in docs] == ["1"]

    def test_embedding_retrieval_rejects_empty_query(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="non-empty list of floats"):
            store._embedding_retrieval(query_embedding=[])

    def test_embedding_retrieval_rejects_wrong_dimensionality(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="has 2 dimensions, but the store is configured for 3"):
            store._embedding_retrieval(query_embedding=[0.1, 0.2])

    def test_client_kwargs_forward_region_and_resolved_credentials(self) -> None:
        store = make_store()
        assert store._client_kwargs() == {
            "region_name": "us-east-1",
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_session_token": "test-session-token",
        }

    def test_client_kwargs_omit_unset_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        store = DynamoDBDocumentStore(embedding_dimension=3)
        # nothing forwarded: boto3 falls back to its default credential and region chain
        assert store._client_kwargs() == {}

    def test_init_does_not_create_a_client(self) -> None:
        store = make_store()
        assert store._client is None
        assert store._async_session is None

    @pytest.mark.parametrize("top_k", [0, -1, SEARCH_VECTORS_MAX_TOP_K + 1])
    def test_embedding_retrieval_rejects_top_k_outside_dynamodb_limit(self, top_k: int) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="top_k must be between 1 and 100"):
            store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=top_k)

    def test_embedding_retrieval_with_filters_fetches_at_most_the_dynamodb_limit(self) -> None:
        """
        `TopK` is capped at 100 by a non-adjustable DynamoDB quota, so over-fetching for
        client-side filtering must never exceed it.
        """
        store = make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {"SearchResults": []}
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            store._embedding_retrieval(
                query_embedding=[0.1, 0.2, 0.3],
                top_k=50,
                filters={"field": "meta.topic", "operator": "==", "value": "ai"},
            )
            _, kwargs = mock_client.search_vectors.call_args
            assert kwargs["TopK"] == SEARCH_VECTORS_MAX_TOP_K

    def test_embedding_retrieval_without_filters_fetches_exactly_top_k(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {"SearchResults": []}
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=7)
            _, kwargs = mock_client.search_vectors.call_args
            assert kwargs["TopK"] == 7

    def test_embedding_retrieval_returns_scored_documents(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {
            "SearchResults": [
                {
                    "Item": {"id": {"S": "1"}, "payload": {"S": '{"content": "hello"}'}},
                    # DynamoDB returns a COSINE *distance* (0 = identical, 2 = opposite).
                    "Score": 0.5,
                }
            ]
        }
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=1)
            assert len(docs) == 1
            assert docs[0].id == "1"
            # distance 0.5 -> similarity 1 - 0.5/2 = 0.75 (Haystack: higher = more relevant)
            assert docs[0].score == pytest.approx(0.75)
            # verify we call SearchVectors with the real API param shape
            _, kwargs = mock_client.search_vectors.call_args
            assert kwargs["SearchVector"] == [{"N": "0.1"}, {"N": "0.2"}, {"N": "0.3"}]
            assert "QueryVector" not in kwargs

    def test_embedding_retrieval_converts_cosine_distance_to_similarity(self) -> None:
        """An identical vector (distance 0) must score 1.0 and an opposite one (distance 2) 0.0."""
        store = make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {
            "SearchResults": [
                {"Item": {"id": {"S": "same"}, "payload": {"S": '{"content": "a"}'}}, "Score": 0.0},
                {"Item": {"id": {"S": "orth"}, "payload": {"S": '{"content": "b"}'}}, "Score": 1.0},
                {"Item": {"id": {"S": "opp"}, "payload": {"S": '{"content": "c"}'}}, "Score": 2.0},
            ]
        }
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store._embedding_retrieval(query_embedding=[1.0, 0.0, 0.0], top_k=3)
            assert [d.score for d in docs] == [
                pytest.approx(1.0),
                pytest.approx(0.5),
                pytest.approx(0.0),
            ]

    def test_embedding_retrieval_applies_client_side_filter(self) -> None:
        store = make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {
            "SearchResults": [
                {
                    "Item": {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}},
                    "Score": 0.9,
                },
                {
                    "Item": {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
                    "Score": 0.8,
                },
            ]
        }
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store._embedding_retrieval(
                query_embedding=[0.1, 0.2, 0.3],
                top_k=5,
                filters={"field": "meta.topic", "operator": "==", "value": "db"},
            )
            assert [d.id for d in docs] == ["2"]

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        store = DynamoDBDocumentStore(
            table_name="test_docs",
            index_name="test_index",
            embedding_dimension=3,
            region_name="us-east-1",
        )
        data = store.to_dict()
        rebuilt = DynamoDBDocumentStore.from_dict(data)
        assert rebuilt.table_name == store.table_name
        assert rebuilt.index_name == store.index_name
        assert rebuilt.embedding_dimension == store.embedding_dimension


@pytest.mark.integration
class TestDynamoDBDocumentStoreIntegration(DocumentStoreBaseExtendedTests):
    """
    Runs against a real DynamoDB table in AWS; see `conftest.py` for the opt-in and the fixtures.
    """

    @pytest.fixture
    def document_store(self, clean_store: DynamoDBDocumentStore) -> DynamoDBDocumentStore:
        return clean_store

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        assert_documents_equal_ignoring_order(received, expected)

    def test_write_documents(self, document_store: DynamoDBDocumentStore) -> None:
        docs = [Document(content="doc1"), Document(content="doc2")]
        assert document_store.write_documents(docs) == 2

    def test_embedding_retrieval_ranks_by_similarity(self, document_store: DynamoDBDocumentStore) -> None:
        """
        Exercises the real `SearchVectors` path end to end: the base suite never calls it.

        Queries with a vector identical to one stored vector and checks ordering plus the
        distance-to-similarity conversion (identical -> 1.0, orthogonal -> 0.5).
        """
        near = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
        mid = [0.7, 0.7] + [0.0] * (EMBEDDING_DIMENSION - 2)
        far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSION - 2)
        document_store.write_documents(
            [
                Document(id="near", content="near", embedding=near),
                Document(id="mid", content="mid", embedding=mid),
                Document(id="far", content="far", embedding=far),
            ]
        )
        results = document_store._embedding_retrieval(query_embedding=near, top_k=3)
        assert [d.id for d in results] == ["near", "mid", "far"]
        assert results[0].score == pytest.approx(1.0, abs=1e-3)
        assert results[2].score == pytest.approx(0.5, abs=1e-3)
        assert results[0].score >= results[1].score >= results[2].score

    def test_embedding_retrieval_applies_filters(self, document_store: DynamoDBDocumentStore) -> None:
        query = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
        document_store.write_documents(
            [
                Document(id=f"{group}-{i}", content=group, meta={"group": group}, embedding=query)
                for group in ("a", "b")
                for i in range(3)
            ]
        )
        results = document_store._embedding_retrieval(
            query_embedding=query, top_k=10, filters={"field": "meta.group", "operator": "==", "value": "b"}
        )
        assert sorted(d.id for d in results) == ["b-0", "b-1", "b-2"]
