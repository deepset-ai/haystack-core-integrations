# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy

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
def test_count_documents_empty(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.list_vectors.return_value = {"vectors": []}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    assert store.count_documents() == 0


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_count_documents_pagination(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.list_vectors.side_effect = [
        {"vectors": [{"key": "1"}, {"key": "2"}], "nextToken": "tok"},
        {"vectors": [{"key": "3"}]},
    ]
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    assert store.count_documents() == 3


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.put_vectors.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = [
        Document(id="1", content="Hello", embedding=[0.1] * 4),
        Document(id="2", content="World", embedding=[0.2] * 4),
    ]
    assert store.write_documents(docs) == 2
    client.put_vectors.assert_called_once()


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_empty(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    assert store.write_documents([]) == 0
    client.put_vectors.assert_not_called()


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_no_embedding_raises(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    with pytest.raises(DocumentStoreError, match="has no embedding"):
        store.write_documents([Document(id="1", content="Hello")])


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_skip_existing(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.get_vectors.return_value = {"vectors": [{"key": "1"}]}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = [Document(id="1", content="Hello", embedding=[0.1] * 4)]
    assert store.write_documents(docs, policy=DuplicatePolicy.SKIP) == 0


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_none_policy_raises(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.get_vectors.return_value = {"vectors": [{"key": "1"}]}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = [Document(id="1", content="Hello", embedding=[0.1] * 4)]
    with pytest.raises(DocumentStoreError, match="already exist"):
        store.write_documents(docs, policy=DuplicatePolicy.NONE)


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_metadata(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.put_vectors.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = [Document(id="1", content="Hello", embedding=[0.1] * 4, meta={"category": "test", "year": 2024})]
    store.write_documents(docs)

    vectors = client.put_vectors.call_args[1]["vectors"]
    assert len(vectors) == 1
    assert vectors[0]["key"] == "1"
    assert vectors[0]["metadata"]["_content"] == "Hello"
    assert vectors[0]["metadata"]["category"] == "test"
    assert vectors[0]["metadata"]["year"] == 2024


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_delete_documents(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.delete_vectors.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store.delete_documents(["1", "2"])
    client.delete_vectors.assert_called_once_with(vectorBucketName="b", indexName="i", keys=["1", "2"])


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_delete_documents_empty(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store.delete_documents([])
    client.delete_vectors.assert_not_called()


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval(mock_boto3):
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
    assert docs[0].score == pytest.approx(0.95)  # 1 - 0.05
    assert docs[0].meta == {"category": "news"}


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_with_filters(mock_boto3):
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


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_empty_embedding_raises(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    with pytest.raises(ValueError, match="non-empty"):
        store._embedding_retrieval(query_embedding=[])


def test_document_to_s3_vector():
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
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    vector = S3VectorsDocumentStore._document_to_s3_vector(doc)
    restored = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert restored.id == doc.id
    assert restored.content == doc.content
    assert restored.embedding == doc.embedding
    assert restored.meta == doc.meta
