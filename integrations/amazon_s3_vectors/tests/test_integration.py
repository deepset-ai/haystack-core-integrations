# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for S3VectorsDocumentStore.

Requires valid AWS credentials with s3vectors permissions.
Run with: hatch run test:integration
"""

import os
import time
import uuid

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


def _aws_credentials_available() -> bool:
    """Check if AWS credentials are available via any mechanism."""
    # Explicit env vars
    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ROLE_ARN"):
        return True
    # Fallback: try to resolve credentials from the default chain
    try:
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


# Guard: skip all integration tests when AWS credentials are not available
pytestmark = pytest.mark.skipif(
    not _aws_credentials_available(),
    reason="AWS credentials not configured",
)

# Use a small dimension to keep payloads light
DIMENSION = 4
REGION = "us-east-1"


def _random_name(prefix: str = "haystack-test") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_doc(doc_id: str, content: str, embedding: list[float] | None = None, meta: dict | None = None) -> Document:
    return Document(
        id=doc_id,
        content=content,
        embedding=embedding or [0.1] * DIMENSION,
        meta=meta or {},
    )


@pytest.fixture(scope="module")
def doc_store():
    """Create a real S3 Vectors document store with a unique bucket + index, tear down after."""
    bucket_name = _random_name("hs-integ")
    index_name = _random_name("idx")

    store = S3VectorsDocumentStore(
        vector_bucket_name=bucket_name,
        index_name=index_name,
        dimension=DIMENSION,
        distance_metric="cosine",
        region_name=REGION,
        create_bucket_and_index=True,
        non_filterable_metadata_keys=[],
    )
    # Force initialization
    store._get_client()

    yield store

    # Cleanup
    client = store._client
    try:
        client.delete_index(vectorBucketName=bucket_name, indexName=index_name)
    except Exception:
        pass
    try:
        client.delete_vector_bucket(vectorBucketName=bucket_name)
    except Exception:
        pass


@pytest.mark.integration
class TestWriteAndCount:
    def test_write_and_count(self, doc_store):
        docs = [
            _make_doc("int-1", "First document"),
            _make_doc("int-2", "Second document"),
            _make_doc("int-3", "Third document"),
        ]
        written = doc_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert written == 3

        # S3 Vectors is eventually consistent — give it a moment
        time.sleep(2)

        count = doc_store.count_documents()
        assert count == 3

    def test_overwrite(self, doc_store):
        doc = _make_doc("int-1", "Updated first document", embedding=[0.9] * DIMENSION)
        written = doc_store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1

    def test_skip_existing(self, doc_store):
        doc = _make_doc("int-1", "Should be skipped")
        written = doc_store.write_documents([doc], policy=DuplicatePolicy.SKIP)
        assert written == 0

    def test_none_policy_raises_on_existing(self, doc_store):
        doc = _make_doc("int-1", "Should fail")
        with pytest.raises(DocumentStoreError, match="already exist"):
            doc_store.write_documents([doc], policy=DuplicatePolicy.NONE)

    def test_write_without_embedding_raises(self, doc_store):
        doc = Document(id="no-emb", content="No embedding")
        with pytest.raises(DocumentStoreError, match="has no embedding"):
            doc_store.write_documents([doc])


@pytest.mark.integration
class TestQuery:
    def test_embedding_retrieval(self, doc_store):
        # Query with an embedding close to [0.1, 0.1, 0.1, 0.1]
        docs = doc_store._embedding_retrieval(
            query_embedding=[0.1] * DIMENSION,
            top_k=10,
        )
        assert len(docs) > 0
        # All returned docs should have a score (cosine similarity)
        for doc in docs:
            assert doc.score is not None
            assert doc.content is not None

    def test_embedding_retrieval_with_metadata_filter(self, doc_store):
        # Write a doc with distinctive metadata
        tagged_doc = _make_doc(
            "int-tagged",
            "Tagged document",
            embedding=[0.5] * DIMENSION,
            meta={"category": "special"},
        )
        doc_store.write_documents([tagged_doc], policy=DuplicatePolicy.OVERWRITE)
        time.sleep(2)

        # Query with filter
        filters = {"field": "meta.category", "operator": "==", "value": "special"}
        docs = doc_store._embedding_retrieval(
            query_embedding=[0.5] * DIMENSION,
            filters=filters,
            top_k=10,
        )
        assert len(docs) >= 1
        assert all(d.meta.get("category") == "special" for d in docs)

    def test_retriever_component(self, doc_store):
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store, top_k=5)
        result = retriever.run(query_embedding=[0.1] * DIMENSION)
        assert "documents" in result
        assert len(result["documents"]) > 0


@pytest.mark.integration
class TestFilterDocuments:
    def test_filter_documents_no_filter(self, doc_store):
        docs = doc_store.filter_documents()
        assert len(docs) > 0
        for doc in docs:
            assert doc.id is not None

    def test_filter_documents_with_filter(self, doc_store):
        filters = {"field": "meta.category", "operator": "==", "value": "special"}
        docs = doc_store.filter_documents(filters=filters)
        assert all(d.meta.get("category") == "special" for d in docs)


@pytest.mark.integration
class TestDelete:
    def test_delete_documents(self, doc_store):
        # Write a doc to delete
        doc = _make_doc("int-to-delete", "Delete me")
        doc_store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        time.sleep(2)

        doc_store.delete_documents(["int-to-delete"])
        time.sleep(2)

        # Verify it's gone by trying to get it
        client = doc_store._client
        response = client.get_vectors(
            vectorBucketName=doc_store.vector_bucket_name,
            indexName=doc_store.index_name,
            keys=["int-to-delete"],
        )
        assert len(response.get("vectors", [])) == 0


@pytest.mark.integration
class TestSerialization:
    def test_roundtrip(self, doc_store):
        d = doc_store.to_dict()
        restored = S3VectorsDocumentStore.from_dict(d)
        assert restored.vector_bucket_name == doc_store.vector_bucket_name
        assert restored.index_name == doc_store.index_name
        assert restored.dimension == doc_store.dimension
        assert restored.distance_metric == doc_store.distance_metric
