# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils import Secret

from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

_MODULE = "haystack_integrations.document_stores.dynamodb.document_store"


def _make_store(**kwargs) -> DynamoDBDocumentStore:
    return DynamoDBDocumentStore(
        table_name="test_docs",
        index_name="test_index",
        embedding_dimension=3,
        region_name="us-east-1",
        aws_access_key_id=Secret.from_token("test-key"),
        aws_secret_access_key=Secret.from_token("test-secret"),
        **kwargs,
    )


def _require_live_aws() -> str:
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region or not os.environ.get("HAYSTACK_DYNAMODB_INTEGRATION_TESTS"):
        pytest.skip("Set AWS_DEFAULT_REGION and HAYSTACK_DYNAMODB_INTEGRATION_TESTS=1 to run integration tests.")
    return region


class TestDynamoDBDocumentStore:
    def test_init_default(self) -> None:
        store = DynamoDBDocumentStore()
        assert store.table_name == "haystack_documents"
        assert store.index_name == "haystack_vector_index"
        assert store.embedding_dimension == 768
        assert store.similarity_function == "cosine"

    def test_init_rejects_non_cosine_similarity(self) -> None:
        with pytest.raises(ValueError, match="Only 'cosine' is supported"):
            DynamoDBDocumentStore(similarity_function="dot_product")

    def test_count_documents(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Count": 5}]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            assert store.count_documents() == 5

    def test_write_documents_rejects_non_document(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            store.write_documents([{"not": "a document"}])  # type: ignore[list-item]

    def test_write_documents_empty_list(self) -> None:
        store = _make_store()
        assert store.write_documents([]) == 0

    def test_write_documents_fail_policy_raises_on_duplicate(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        mock_client.get_item.return_value = {"Item": {"id": {"S": "1"}}}
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            with pytest.raises(DuplicateDocumentError):
                store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.FAIL)

    def test_write_documents_skip_policy_skips_duplicate(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        mock_client.get_item.return_value = {"Item": {"id": {"S": "1"}}}
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            written = store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)
            assert written == 0
            mock_client.put_item.assert_not_called()

    def test_write_documents_overwrite_policy_writes_regardless(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        mock_client.get_item.return_value = {"Item": {"id": {"S": "1"}}}
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            written = store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.OVERWRITE)
            assert written == 1
            mock_client.put_item.assert_called_once()

    def test_delete_documents_empty_list_is_noop(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        with patch.object(store, "_get_client", return_value=mock_client):
            store.delete_documents([])
            mock_client.delete_item.assert_not_called()

    def test_delete_documents_calls_delete_item_per_id(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            store.delete_documents(["1", "2"])
            assert mock_client.delete_item.call_count == 2

    def test_filter_documents_no_filter_returns_all(self) -> None:
        store = _make_store()
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
        store = _make_store()
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
        store = _make_store()
        with pytest.raises(ValueError, match="non-empty list of floats"):
            store._embedding_retrieval(query_embedding=[])

    def test_embedding_retrieval_returns_scored_documents(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {
            "Vectors": [
                {
                    "Item": {"id": {"S": "1"}, "payload": {"S": '{"content": "hello"}'}},
                    "Distance": 0.95,
                }
            ]
        }
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=1)
            assert len(docs) == 1
            assert docs[0].id == "1"
            assert docs[0].score == 0.95

    def test_embedding_retrieval_applies_client_side_filter(self) -> None:
        store = _make_store()
        mock_client = MagicMock()
        mock_client.search_vectors.return_value = {
            "Vectors": [
                {
                    "Item": {"id": {"S": "1"}, "payload": {"S": '{"content": "a", "meta": {"topic": "ai"}}'}},
                    "Distance": 0.9,
                },
                {
                    "Item": {"id": {"S": "2"}, "payload": {"S": '{"content": "b", "meta": {"topic": "db"}}'}},
                    "Distance": 0.8,
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
class TestDynamoDBDocumentStoreIntegration(DocumentStoreBaseTests):
    """
    Runs against a real DynamoDB table in AWS. Skipped by default; requires
    AWS_DEFAULT_REGION and HAYSTACK_DYNAMODB_INTEGRATION_TESTS=1, plus real AWS
    credentials on the calling environment (never hardcoded here).
    """

    def test_write_documents(self, document_store: DynamoDBDocumentStore) -> None:
        docs = [Document(content="doc1"), Document(content="doc2")]
        assert document_store.write_documents(docs) == 2

    @pytest.fixture
    def document_store(self, request: pytest.FixtureRequest) -> DynamoDBDocumentStore:
        _require_live_aws()
        # A random suffix (not just the deterministic test name) prevents this run's
        # table from colliding with an orphaned table of the same name left behind by
        # an earlier failed/interrupted run — which surfaced as spurious
        # DuplicateDocumentError failures against pre-existing leftover items during
        # real-AWS validation.
        unique_suffix = uuid.uuid4().hex[:8]
        store = DynamoDBDocumentStore(
            table_name=f"haystack_test_{request.node.name}_{unique_suffix}",
            index_name="test_index",
            embedding_dimension=768,
        )
        yield store
        with contextlib.suppress(Exception):
            store._get_client().delete_table(TableName=store.table_name)
