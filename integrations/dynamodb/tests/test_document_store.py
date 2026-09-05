# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import time
import uuid
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
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


def _best_effort_delete_table(store: DynamoDBDocumentStore) -> None:
    """
    Best-effort table delete for per-test teardown.

    A freshly-written table often still has its vector index in a CREATING/UPDATING state
    when the test finishes, and DynamoDB rejects `DeleteTable` during that window with
    `ResourceInUseException`. We therefore attempt the delete but never block or fail the
    test on it — any table that can't be deleted yet is swept later by the session-scoped
    `_cleanup_test_tables` fixture, once its index has finished building.
    """
    with contextlib.suppress(Exception):
        store._get_client().delete_table(TableName=store.table_name)


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
            "SearchResults": [
                {
                    "Item": {"id": {"S": "1"}, "payload": {"S": '{"content": "hello"}'}},
                    "Score": 0.95,
                }
            ]
        }
        with patch.object(store, "_get_client", return_value=mock_client):
            store._table_ready = True
            docs = store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=1)
            assert len(docs) == 1
            assert docs[0].id == "1"
            assert docs[0].score == 0.95
            # verify we call SearchVectors with the real API param shape
            _, kwargs = mock_client.search_vectors.call_args
            assert kwargs["SearchVector"] == [{"N": "0.1"}, {"N": "0.2"}, {"N": "0.3"}]
            assert "QueryVector" not in kwargs

    def test_embedding_retrieval_applies_client_side_filter(self) -> None:
        store = _make_store()
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
class TestDynamoDBDocumentStoreIntegration(DocumentStoreBaseTests):
    """
    Runs against a real DynamoDB table in AWS. Skipped by default; requires
    AWS_DEFAULT_REGION and HAYSTACK_DYNAMODB_INTEGRATION_TESTS=1, plus real AWS
    credentials on the calling environment (never hardcoded here).
    """

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        """
        Compares two lists of Documents order-independently.

        `filter_documents` has no ordering contract, and DynamoDB's `Scan`/`SearchVectors`
        return items in an order that does not match the base suite's insertion order. We
        therefore sort both sides by `id` before comparing. We also null the `score` (set on
        retrieval, non-deterministic) and compare embeddings approximately, since floats do
        not survive the DynamoDB number round-trip exactly. This mirrors the approach used by
        the merged `opensearch` integration's document-store tests.
        """
        assert len(received) == len(expected)
        received = sorted(received, key=lambda x: x.id)
        expected = sorted(expected, key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            received_doc.score = None
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)
            received_doc.embedding, expected_doc.embedding = None, None
            assert received_doc == expected_doc

    def test_write_documents(self, document_store: DynamoDBDocumentStore) -> None:
        docs = [Document(content="doc1"), Document(content="doc2")]
        assert document_store.write_documents(docs) == 2

    def test_embedding_retrieval_ranks_by_similarity(self) -> None:
        """
        Exercises the real `SearchVectors` vector-search path end to end.

        The base `DocumentStoreBaseTests` suite never calls the embedding-retrieval path, so
        this test is what actually validates the native vector search against real AWS: it
        writes docs with known embeddings, queries with a vector identical to one of them, and
        asserts that doc ranks first with the highest score (COSINE: higher score == closer).
        """
        _require_live_aws()
        dim = 8
        store = DynamoDBDocumentStore(
            table_name=f"haystack_test_embedding_retrieval_{uuid.uuid4().hex[:8]}",
            index_name="test_index",
            embedding_dimension=dim,
        )
        try:
            near = [1.0] + [0.0] * (dim - 1)
            mid = [0.7, 0.7] + [0.0] * (dim - 2)
            far = [0.0, 1.0] + [0.0] * (dim - 2)
            store.write_documents(
                [
                    Document(id="near", content="near", embedding=near),
                    Document(id="mid", content="mid", embedding=mid),
                    Document(id="far", content="far", embedding=far),
                ]
            )
            results = store._embedding_retrieval(query_embedding=near, top_k=3)
            assert [d.id for d in results] == ["near", "mid", "far"]
            assert all(d.score is not None for d in results)
            assert results[0].score >= results[1].score >= results[2].score
        finally:
            _best_effort_delete_table(store)

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
        _best_effort_delete_table(store)

    @pytest.fixture(scope="class", autouse=True)
    def _cleanup_test_tables(self) -> None:
        """
        Session-safety net: after all tests in this class finish, sweep any leftover
        `haystack_test_*` tables whose per-test best-effort delete was rejected because their
        vector index was still building. By this point the indexes have settled, so these
        deletes succeed and the run leaves the AWS account clean. Retries briefly to ride out
        any tables still transitioning.
        """
        yield
        region = os.environ.get("AWS_DEFAULT_REGION")
        if not region or not os.environ.get("HAYSTACK_DYNAMODB_INTEGRATION_TESTS"):
            return
        client = boto3.client("dynamodb", region_name=region)
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            leftovers = [t for t in client.list_tables().get("TableNames", []) if t.startswith("haystack_test_")]
            if not leftovers:
                break
            still_pending = False
            for table_name in leftovers:
                try:
                    client.delete_table(TableName=table_name)
                except ClientError as e:
                    code = e.response["Error"]["Code"]
                    if code == "ResourceInUseException":
                        still_pending = True
                    elif code != "ResourceNotFoundException":
                        raise
            if not still_pending:
                break
            time.sleep(10)
