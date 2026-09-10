# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for the integration tests, which run against real AWS.

There is no local DynamoDB emulator with vector-index support, so the integration tests are
skipped unless `AWS_DEFAULT_REGION` and `HAYSTACK_DYNAMODB_INTEGRATION_TESTS=1` are set and boto3
can find credentials.
"""

import contextlib
import dataclasses
import os
import time
import uuid
from collections.abc import Iterator

import boto3
import pytest
from botocore.exceptions import ClientError
from haystack.dataclasses import Document

from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

# Matches the embedding size used by haystack's `FilterableDocsFixtureMixin`.
EMBEDDING_DIMENSION = 768

# One prefix per test process. Several CI matrix jobs share the AWS account and run concurrently,
# so the final sweep must only ever touch tables created by this very run.
TABLE_PREFIX = f"haystack_test_{uuid.uuid4().hex[:8]}_"


def live_aws_region() -> str | None:
    """Returns the region to test against, or `None` when the live tests are not opted in."""
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region or not os.environ.get("HAYSTACK_DYNAMODB_INTEGRATION_TESTS"):
        return None
    return region


def require_live_aws() -> str:
    region = live_aws_region()
    if region is None:
        pytest.skip("Set AWS_DEFAULT_REGION and HAYSTACK_DYNAMODB_INTEGRATION_TESTS=1 to run integration tests.")
    return region


def _best_effort_delete_table(store: DynamoDBDocumentStore) -> None:
    """
    DynamoDB rejects `DeleteTable` with `ResourceInUseException` while a table's vector index is
    still transitioning, so the delete is attempted but never blocks or fails a test. Whatever is
    left over is removed by `_sweep_test_tables` once the indexes have settled.
    """
    with contextlib.suppress(Exception):
        store._get_client().delete_table(TableName=store.table_name)


@pytest.fixture(scope="class")
def live_store(request: pytest.FixtureRequest) -> Iterator[DynamoDBDocumentStore]:
    """
    One real table per test class.

    Creating a table and waiting for its vector index takes 20-60 s, so it is paid once per
    class; `document_store` empties the table before every test instead.
    """
    require_live_aws()
    store = DynamoDBDocumentStore(
        table_name=f"{TABLE_PREFIX}{request.node.name}",
        index_name="test_index",
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    store._ensure_table()
    yield store
    _best_effort_delete_table(store)


@pytest.fixture
def clean_store(live_store: DynamoDBDocumentStore) -> DynamoDBDocumentStore:
    """
    The class table, emptied before the test.

    Test classes expose this as their `document_store` fixture: haystack's test mixins define a
    `document_store` fixture of their own that raises `NotImplementedError`, so a class-level
    override is required and a conftest fixture of that name would be shadowed.
    """
    live_store.delete_all_documents()
    return live_store


def assert_documents_equal_ignoring_order(received: list[Document], expected: list[Document]) -> None:
    """
    Order-independent document comparison for the haystack mixins.

    `filter_documents` has no ordering contract, and DynamoDB's `Scan`/`SearchVectors` return
    items in an order that does not match the base suite's insertion order, so both sides are
    sorted by `id`. `score` is nulled (set on retrieval) and embeddings are compared
    approximately, since floats do not survive the DynamoDB number round-trip exactly.
    """
    assert len(received) == len(expected)
    received = sorted(received, key=lambda x: x.id)
    expected = sorted(expected, key=lambda x: x.id)
    for received_doc, expected_doc in zip(received, expected, strict=True):
        if received_doc.embedding is None:
            assert expected_doc.embedding is None
        else:
            assert received_doc.embedding == pytest.approx(expected_doc.embedding)
        # `dataclasses.replace` rather than in-place mutation: Haystack warns that mutating a
        # `Document` can affect other users of the same instance.
        assert dataclasses.replace(received_doc, score=None, embedding=None) == dataclasses.replace(
            expected_doc, score=None, embedding=None
        )


@pytest.fixture(scope="session", autouse=True)
def _sweep_test_tables() -> Iterator[None]:
    """
    Safety net: after the whole session, delete any table of this run whose per-class delete was
    rejected while its index was still settling. Retries for a few minutes to ride out tables
    that are still transitioning. Only tables carrying this run's prefix are touched.
    """
    yield
    region = live_aws_region()
    if region is None:
        return
    client = boto3.client("dynamodb", region_name=region)
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        leftovers = [t for t in client.list_tables().get("TableNames", []) if t.startswith(TABLE_PREFIX)]
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
            return
        time.sleep(10)
