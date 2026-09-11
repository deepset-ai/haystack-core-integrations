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
import warnings
from collections.abc import Iterator

import boto3
import pytest
from botocore.exceptions import ClientError
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

# Matches the embedding size used by haystack's `FilterableDocsFixtureMixin`.
EMBEDDING_DIMENSION = 768

# One prefix per test process. Several CI matrix jobs share the AWS account and run concurrently,
# so the final sweep must only ever touch tables created by this very run.
TABLE_PREFIX = f"haystack_test_{uuid.uuid4().hex[:8]}_"


def make_store(**kwargs) -> DynamoDBDocumentStore:
    """
    A store with static credentials for unit tests; never talks to AWS unless a client is used.

    All three credential Secrets are fixed tokens so the tests do not depend on `AWS_*` variables
    in the environment (CI exports `AWS_SESSION_TOKEN` after assuming the OIDC role).
    """
    return DynamoDBDocumentStore(
        table_name="test_docs",
        index_name="test_index",
        embedding_dimension=3,
        region_name="us-east-1",
        aws_access_key_id=Secret.from_token("test-key"),
        aws_secret_access_key=Secret.from_token("test-secret"),
        aws_session_token=Secret.from_token("test-session-token"),
        **kwargs,
    )


def table_description(
    *,
    index_name: str = "test_index",
    dimensions: int = 3,
    distance_function: str = "COSINE",
    vector_attribute: str = "embedding",
    index_status: str = "ACTIVE",
    backfilling: bool | None = None,
    key_schema: list[dict[str, str]] | None = None,
) -> dict:
    """Builds a `DescribeTable` payload shaped like the one for a table created by the store."""
    index: dict = {
        "IndexName": index_name,
        "Dimensions": dimensions,
        "DistanceFunction": distance_function,
        "VectorAttribute": {"AttributeName": vector_attribute},
        "IndexStatus": index_status,
    }
    if backfilling is not None:
        index["Backfilling"] = backfilling
    return {
        "Table": {
            "TableName": "test_docs",
            "TableStatus": "ACTIVE",
            "KeySchema": key_schema if key_schema is not None else [{"AttributeName": "id", "KeyType": "HASH"}],
            "VectorIndexes": [index],
        }
    }


def client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


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

    Never fails the session: a sweep that cannot run (for example because the credentials lack
    `ListTables`) is reported as a warning so the test results stay readable.
    """
    yield
    region = live_aws_region()
    if region is None:
        return
    client = boto3.client("dynamodb", region_name=region)
    deadline = time.monotonic() + 300.0
    try:
        while time.monotonic() < deadline:
            leftovers = [t for t in client.list_tables().get("TableNames", []) if t.startswith(TABLE_PREFIX)]
            still_pending = False
            for table_name in leftovers:
                try:
                    client.delete_table(TableName=table_name)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ResourceInUseException":
                        still_pending = True
                    elif e.response["Error"]["Code"] != "ResourceNotFoundException":
                        raise
            if not still_pending:
                return
            time.sleep(10)
    except ClientError as e:
        warnings.warn(f"Could not sweep leftover {TABLE_PREFIX}* tables: {e}", stacklevel=1)
