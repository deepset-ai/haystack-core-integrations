# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    WriteDocumentsTest,
)

from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


def test_init_is_lazy():
    store = S3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
    )
    assert store._client is None


def test_init_default_params():
    store = S3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
    )
    assert store.vector_bucket_name == "my-bucket"
    assert store.index_name == "my-index"
    assert store.dimension == 768
    assert store.distance_metric == "cosine"
    assert store.create_bucket_and_index is True
    assert store.non_filterable_metadata_keys == []


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_to_dict(_mock_boto3):
    store = S3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
        distance_metric="euclidean",
        region_name="us-west-2",
        create_bucket_and_index=False,
    )
    d = store.to_dict()
    assert d == {
        "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
        "init_parameters": {
            "vector_bucket_name": "my-bucket",
            "index_name": "my-index",
            "dimension": 768,
            "distance_metric": "euclidean",
            "region_name": "us-west-2",
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "create_bucket_and_index": False,
            "non_filterable_metadata_keys": [],
        },
    }


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_from_dict(_mock_boto3):
    data = {
        "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
        "init_parameters": {
            "vector_bucket_name": "my-bucket",
            "index_name": "my-index",
            "dimension": 768,
            "distance_metric": "euclidean",
            "region_name": "us-west-2",
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "create_bucket_and_index": False,
            "non_filterable_metadata_keys": [],
        },
    }
    store = S3VectorsDocumentStore.from_dict(data)
    assert store.vector_bucket_name == "my-bucket"
    assert store.index_name == "my-index"
    assert store.dimension == 768
    assert store.distance_metric == "euclidean"
    assert store.region_name == "us-west-2"
    assert store.create_bucket_and_index is False


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_missing_embedding_uses_placeholder(mock_boto3):
    """Every S3 Vectors record needs a vector, so embedding-less documents get the placeholder."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = [Document(id="1", content="ok", embedding=[0.1] * 4), Document(id="2", content="missing")]
    assert store.write_documents(docs) == 2

    written = client.put_vectors.call_args.kwargs["vectors"]
    assert written[0]["data"]["float32"] == [0.1] * 4
    # Non-zero: a cosine index rejects zero-norm vectors.
    assert written[1]["data"]["float32"] == [-10.0] * 4
    # The caller's Document is not mutated.
    assert docs[1].embedding is None


@pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.SKIP])
@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_unsupported_policy_overwrites(mock_boto3, policy):
    """
    put_vectors is an upsert and S3 Vectors has no cheap existence check, so policies other than
    OVERWRITE are ignored instead of paying for an extra read pass.
    """
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    assert store.write_documents([Document(id="1", content="Hello", embedding=[0.1] * 4)], policy=policy) == 1
    client.get_vectors.assert_not_called()
    client.put_vectors.assert_called_once()


def test_write_documents_invalid_input_raises():
    """write_documents must reject non-list / non-Document input before doing anything."""
    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False)
    with pytest.raises(ValueError):
        store.write_documents("not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.write_documents(["not a document"])  # type: ignore[list-item]


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_score_conversion(mock_boto3):
    """Tests our distance-to-score conversion logic — the only non-trivial transform in retrieval."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {
        "vectors": [{"key": "1", "distance": 0.05, "metadata": {"_content": "Hello", "category": "news"}}],
        "distanceMetric": "cosine",
    }
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = store._embedding_retrieval(query_embedding=[0.1] * 4, top_k=5)
    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].content == "Hello"
    assert docs[0].score == pytest.approx(0.95)  # cosine: 1.0 - 0.05
    assert docs[0].meta == {"category": "news"}


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_euclidean_score(mock_boto3):
    """Tests euclidean distance-to-score conversion (negated)."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {
        "vectors": [{"key": "1", "distance": 1.5, "metadata": {}}],
        "distanceMetric": "euclidean",
    }
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, distance_metric="euclidean", region_name="us-east-1"
    )
    docs = store._embedding_retrieval(query_embedding=[0.1] * 4, top_k=5)
    assert docs[0].score == pytest.approx(-1.5)  # euclidean: negated


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_passes_filters(mock_boto3):
    """Tests that Haystack filters are converted and passed to query_vectors."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]}
    store._embedding_retrieval(query_embedding=[0.1] * 4, filters=filters, top_k=5)

    call_args = client.query_vectors.call_args[1]
    assert call_args["filter"] == {"$and": [{"category": {"$eq": "news"}}]}


def test_embedding_retrieval_empty_embedding_raises():
    """Tests our input validation — no mocking needed."""
    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False)
    with pytest.raises(ValueError, match="non-empty"):
        store._embedding_retrieval(query_embedding=[])


def test_document_to_s3_vector():
    """Tests our Document → S3 vector conversion (pure function)."""
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    result = S3VectorsDocumentStore._document_to_s3_vector(doc)
    assert result["key"] == "test-1"
    assert result["data"] == {"float32": [0.1, 0.2, 0.3]}
    assert result["metadata"]["_content"] == "Hello world"
    assert result["metadata"]["category"] == "test"
    assert result["metadata"]["year"] == 2024


def test_s3_vector_to_document():
    """Tests our S3 vector → Document conversion (pure function)."""
    vector = {
        "key": "test-1",
        "data": {"float32": [0.1, 0.2, 0.3]},
        "metadata": {"_content": "Hello world", "category": "test"},
    }
    doc = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert doc.id == "test-1"
    assert doc.content == "Hello world"
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.meta == {"category": "test"}


def test_document_roundtrip():
    """Tests Document → S3 vector → Document is lossless."""
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    vector = S3VectorsDocumentStore._document_to_s3_vector(doc)
    restored = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert restored.id == doc.id
    assert restored.content == doc.content
    assert restored.embedding == doc.embedding
    assert restored.meta == doc.meta


def test_document_roundtrip_without_embedding():
    """A document written without an embedding must read back without one, not with the placeholder."""
    placeholder = [-10.0] * 3
    doc = Document(id="test-1", content="Hello world", meta={"category": "test"})

    vector = S3VectorsDocumentStore._document_to_s3_vector(doc, fallback_embedding=placeholder)
    assert vector["data"]["float32"] == placeholder

    restored = S3VectorsDocumentStore._s3_vector_to_document(vector, placeholder_embedding=placeholder)
    assert restored.embedding is None
    assert restored.content == doc.content
    assert restored.meta == doc.meta


def test_s3_vector_to_document_keeps_embedding_matching_no_placeholder():
    """A real embedding is never mistaken for the placeholder."""
    vector = {"key": "test-1", "data": {"float32": [0.1, 0.2, 0.3]}, "metadata": {}}
    doc = S3VectorsDocumentStore._s3_vector_to_document(vector, placeholder_embedding=[-10.0] * 3)
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    WriteDocumentsTest,
    DeleteDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    UpdateByFilterTest,
    FilterableDocsFixtureMixin,
):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        """
        Compare documents while tolerating two S3 Vectors quirks:

        * embeddings round-trip through float32 storage, so we use `pytest.approx`;
        * the `score` field is not set by `filter_documents`, only by retrieval, so
          we ignore it for equality.
        """
        assert len(received) == len(expected)
        received.sort(key=lambda d: d.id)
        expected.sort(key=lambda d: d.id)
        for r, e in zip(received, expected, strict=True):
            r_norm = replace(r, embedding=None, score=None)
            e_norm = replace(e, embedding=None, score=None)
            assert r_norm == e_norm
            if r.embedding is not None and e.embedding is not None:
                assert r.embedding == pytest.approx(e.embedding, abs=1e-5)

    def test_write_documents(self, document_store: S3VectorsDocumentStore) -> None:
        """
        Default behaviour is OVERWRITE (S3 Vectors `put_vectors` is upsert), so
        writing the same document twice succeeds and returns 1 each time.
        """
        docs = [Document(id="1", content="hello", embedding=[0.1] * 768)]
        assert document_store.write_documents(docs) == 1
        assert document_store.write_documents(docs) == 1

    def test_write_documents_without_embedding_reads_back_as_none(self, document_store: S3VectorsDocumentStore) -> None:
        """
        S3 Vectors stores a placeholder vector for embedding-less documents; `filter_documents`
        must strip it again rather than hand back a vector the caller never supplied.

        `assert_documents_are_equal` ignores embeddings, so this needs its own assertion.
        """
        document_store.write_documents([Document(id="no-emb", content="No embedding here")])

        docs = document_store.filter_documents()
        assert len(docs) == 1
        assert docs[0].content == "No embedding here"
        assert docs[0].embedding is None

    @pytest.mark.skip(reason="S3 Vectors put_vectors is an upsert; only DuplicatePolicy.OVERWRITE is supported")
    def test_write_documents_duplicate_fail(self, document_store: S3VectorsDocumentStore) -> None: ...

    @pytest.mark.skip(reason="S3 Vectors put_vectors is an upsert; only DuplicatePolicy.OVERWRITE is supported")
    def test_write_documents_duplicate_skip(self, document_store: S3VectorsDocumentStore) -> None: ...
