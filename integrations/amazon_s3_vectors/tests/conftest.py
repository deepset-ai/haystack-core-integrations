# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest fixtures for the Amazon S3 Vectors integration tests.

The `document_store` fixture used by the Haystack base test mixins
(`CountDocumentsTest`, `WriteDocumentsTest`, `DeleteDocumentsTest`,
`FilterDocumentsTest`) is defined here.

A single vector bucket is created per test session (creating buckets is the
slow part). Each test gets its own vector index inside that shared bucket so
state is isolated.

Two quirks of S3 Vectors are smoothed over here so the generic base tests
can run unchanged:

1. **Embeddings are required.** The base tests write `Document(content="...")`
   without an embedding. We wrap `write_documents` so any document missing an
   embedding gets a deterministic zero vector of the right dimension.
2. **Writes are eventually consistent.** We sleep briefly after `write_documents`
   and `delete_documents` so the subsequent `filter_documents`/`count_documents`
   reflects the new state.
"""

from __future__ import annotations

import os
import time
import uuid
import warnings
from collections.abc import Iterator

import boto3
import pytest
from botocore.exceptions import ClientError
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore

# Dimension used by Haystack's `FilterableDocsFixtureMixin` test corpus.
DIMENSION = 768
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Eventual-consistency budget for S3 Vectors write/delete propagation.
WRITE_SLEEP_SECONDS = 5
DELETE_SLEEP_SECONDS = 5


def _aws_credentials_available() -> bool:
    """Return True if any boto3 credential source is configured."""
    if any(os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_ROLE_ARN")):
        return True
    try:
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Skip every integration test when AWS credentials are not configured."""
    if _aws_credentials_available():
        return
    skip = pytest.mark.skip(reason="AWS credentials not configured")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def s3_vectors_bucket() -> Iterator[str]:
    """Create one vector bucket for the test session, tear it down at the end."""
    if not _aws_credentials_available():
        pytest.skip("AWS credentials not configured")

    bucket_name = f"haystack-test-{uuid.uuid4().hex[:8]}"
    client = boto3.client("s3vectors", region_name=REGION)
    client.create_vector_bucket(vectorBucketName=bucket_name)

    yield bucket_name

    # Tear down: delete every index in the bucket, then the bucket itself.
    try:
        next_token: str | None = None
        while True:
            kwargs: dict = {"vectorBucketName": bucket_name}
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_indexes(**kwargs)
            for idx in response.get("indexes", []):
                try:
                    client.delete_index(vectorBucketName=bucket_name, indexName=idx["indexName"])
                except ClientError:
                    pass
            next_token = response.get("nextToken")
            if not next_token:
                break
    except ClientError:
        pass

    try:
        client.delete_vector_bucket(vectorBucketName=bucket_name)
    except ClientError:
        pass


@pytest.fixture
def document_store(s3_vectors_bucket: str) -> Iterator[S3VectorsDocumentStore]:
    """
    Provide a fresh S3VectorsDocumentStore (one per test) with `write_documents`
    and `delete_documents` wrapped to:

    * inject a default embedding for any `Document` that doesn't have one, and
    * sleep briefly afterwards to absorb S3 Vectors' eventual consistency.
    """
    index_name = f"idx-{uuid.uuid4().hex[:10]}"

    store = S3VectorsDocumentStore(
        vector_bucket_name=s3_vectors_bucket,
        index_name=index_name,
        dimension=DIMENSION,
        distance_metric="cosine",
        region_name=REGION,
        create_bucket_and_index=True,
        non_filterable_metadata_keys=[],
    )

    # Eagerly create the index so the first write doesn't race with index creation.
    store._get_client()

    original_write = store.write_documents

    def write_with_defaults(documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE) -> int:
        # Only mutate well-formed input; bad input must surface as ValueError from
        # the production code, not crash this wrapper.
        if isinstance(documents, list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for d in documents:
                    if isinstance(d, Document) and d.embedding is None:
                        d.embedding = [0.0] * DIMENSION
        result = original_write(documents, policy)
        time.sleep(WRITE_SLEEP_SECONDS)
        return result

    store.write_documents = write_with_defaults  # type: ignore[method-assign]

    original_delete = store.delete_documents

    def delete_with_sleep(document_ids: list[str]) -> None:
        original_delete(document_ids)
        time.sleep(DELETE_SLEEP_SECONDS)

    store.delete_documents = delete_with_sleep  # type: ignore[method-assign]

    yield store

    # Clean up: drop the per-test index. Best-effort.
    try:
        store._get_client().delete_index(vectorBucketName=s3_vectors_bucket, indexName=index_name)
    except ClientError:
        pass
