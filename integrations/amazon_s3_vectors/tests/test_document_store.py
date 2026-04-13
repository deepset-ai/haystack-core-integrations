# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.list_vectors.return_value = {"vectors": []}
    client.put_vectors.return_value = {}
    client.delete_vectors.return_value = {}
    client.get_vectors.return_value = {"vectors": []}
    client.query_vectors.return_value = {"vectors": []}
    return client


@pytest.fixture
def doc_store(mock_client):
    with patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_client
        store = S3VectorsDocumentStore(
            vector_bucket_name="test-bucket",
            index_name="test-index",
            dimension=768,
            region_name="us-east-1",
        )
        # Force initialization
        store._get_client()
        return store


class TestInit:
    def test_default_params(self):
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


class TestSerialization:
    def test_to_dict(self):
        store = S3VectorsDocumentStore(
            vector_bucket_name="my-bucket",
            index_name="my-index",
            dimension=768,
            distance_metric="euclidean",
            region_name="us-west-2",
            create_bucket_and_index=False,
        )
        d = store.to_dict()
        assert d["init_parameters"]["vector_bucket_name"] == "my-bucket"
        assert d["init_parameters"]["index_name"] == "my-index"
        assert d["init_parameters"]["dimension"] == 768
        assert d["init_parameters"]["distance_metric"] == "euclidean"
        assert d["init_parameters"]["region_name"] == "us-west-2"
        assert d["init_parameters"]["create_bucket_and_index"] is False

    def test_from_dict(self):
        store = S3VectorsDocumentStore(
            vector_bucket_name="my-bucket",
            index_name="my-index",
            dimension=768,
        )
        d = store.to_dict()
        restored = S3VectorsDocumentStore.from_dict(d)
        assert restored.vector_bucket_name == "my-bucket"
        assert restored.index_name == "my-index"
        assert restored.dimension == 768


class TestCountDocuments:
    def test_empty_index(self, doc_store, mock_client):
        mock_client.list_vectors.return_value = {"vectors": []}
        assert doc_store.count_documents() == 0

    def test_with_documents(self, doc_store, mock_client):
        mock_client.list_vectors.return_value = {
            "vectors": [{"key": "1"}, {"key": "2"}, {"key": "3"}],
        }
        assert doc_store.count_documents() == 3

    def test_pagination(self, doc_store, mock_client):
        mock_client.list_vectors.side_effect = [
            {"vectors": [{"key": "1"}, {"key": "2"}], "nextToken": "token1"},
            {"vectors": [{"key": "3"}]},
        ]
        assert doc_store.count_documents() == 3


class TestWriteDocuments:
    def test_write_documents(self, doc_store, mock_client):
        docs = [
            Document(id="1", content="Hello", embedding=[0.1] * 768),
            Document(id="2", content="World", embedding=[0.2] * 768),
        ]
        result = doc_store.write_documents(docs)
        assert result == 2
        mock_client.put_vectors.assert_called_once()

    def test_write_empty_list(self, doc_store, mock_client):
        result = doc_store.write_documents([])
        assert result == 0
        mock_client.put_vectors.assert_not_called()

    def test_write_without_embedding_raises(self, doc_store):
        docs = [Document(id="1", content="Hello")]
        with pytest.raises(DocumentStoreError, match="has no embedding"):
            doc_store.write_documents(docs)

    def test_write_with_metadata(self, doc_store, mock_client):
        docs = [
            Document(
                id="1",
                content="Hello",
                embedding=[0.1] * 768,
                meta={"category": "test", "year": 2024},
            )
        ]
        doc_store.write_documents(docs)
        call_args = mock_client.put_vectors.call_args
        vectors = call_args[1]["vectors"]
        assert len(vectors) == 1
        assert vectors[0]["key"] == "1"
        assert vectors[0]["metadata"]["_content"] == "Hello"
        assert vectors[0]["metadata"]["category"] == "test"
        assert vectors[0]["metadata"]["year"] == 2024

    def test_write_skip_existing(self, doc_store, mock_client):
        mock_client.get_vectors.return_value = {"vectors": [{"key": "1"}]}
        docs = [Document(id="1", content="Hello", embedding=[0.1] * 768)]
        result = doc_store.write_documents(docs, policy=DuplicatePolicy.SKIP)
        assert result == 0
        mock_client.put_vectors.assert_not_called()

    def test_write_none_policy_existing_raises(self, doc_store, mock_client):
        mock_client.get_vectors.return_value = {"vectors": [{"key": "1"}]}
        docs = [Document(id="1", content="Hello", embedding=[0.1] * 768)]
        with pytest.raises(DocumentStoreError, match="already exist"):
            doc_store.write_documents(docs, policy=DuplicatePolicy.NONE)


class TestDeleteDocuments:
    def test_delete(self, doc_store, mock_client):
        doc_store.delete_documents(["1", "2"])
        mock_client.delete_vectors.assert_called_once_with(
            vectorBucketName="test-bucket",
            indexName="test-index",
            keys=["1", "2"],
        )

    def test_delete_empty_list(self, doc_store, mock_client):
        doc_store.delete_documents([])
        mock_client.delete_vectors.assert_not_called()


class TestFilterDocuments:
    def test_no_filters(self, doc_store, mock_client):
        mock_client.list_vectors.return_value = {
            "vectors": [
                {
                    "key": "1",
                    "data": {"float32": [0.1] * 768},
                    "metadata": {"_content": "Hello", "category": "news"},
                }
            ]
        }
        docs = doc_store.filter_documents()
        assert len(docs) == 1
        assert docs[0].id == "1"
        assert docs[0].content == "Hello"
        assert docs[0].meta == {"category": "news"}

    def test_with_filters(self, doc_store, mock_client):
        mock_client.list_vectors.return_value = {
            "vectors": [
                {
                    "key": "1",
                    "data": {"float32": [0.1] * 768},
                    "metadata": {"_content": "Hello", "category": "news"},
                },
                {
                    "key": "2",
                    "data": {"float32": [0.2] * 768},
                    "metadata": {"_content": "World", "category": "sports"},
                },
            ]
        }
        filters = {"field": "meta.category", "operator": "==", "value": "news"}
        docs = doc_store.filter_documents(filters=filters)
        assert len(docs) == 1
        assert docs[0].id == "1"


class TestEmbeddingRetrieval:
    def test_basic_retrieval(self, doc_store, mock_client):
        mock_client.query_vectors.return_value = {
            "vectors": [
                {
                    "key": "1",
                    "distance": 0.05,
                    "metadata": {"_content": "Hello", "category": "news"},
                }
            ],
            "distanceMetric": "cosine",
        }
        docs = doc_store._embedding_retrieval(
            query_embedding=[0.1] * 768,
            top_k=5,
        )
        assert len(docs) == 1
        assert docs[0].id == "1"
        assert docs[0].content == "Hello"
        # cosine distance 0.05 → similarity score 0.95
        assert docs[0].score == pytest.approx(0.95)
        assert docs[0].meta == {"category": "news"}

    def test_retrieval_with_filters(self, doc_store, mock_client):
        mock_client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
            ],
        }
        doc_store._embedding_retrieval(
            query_embedding=[0.1] * 768,
            filters=filters,
            top_k=5,
        )
        call_args = mock_client.query_vectors.call_args[1]
        assert "filter" in call_args
        assert call_args["filter"] == {"$and": [{"category": {"$eq": "news"}}]}

    def test_empty_embedding_raises(self, doc_store):
        with pytest.raises(ValueError, match="non-empty"):
            doc_store._embedding_retrieval(query_embedding=[])


class TestDocumentConversion:
    def test_document_to_s3_vector(self):
        doc = Document(
            id="test-1",
            content="Hello world",
            embedding=[0.1, 0.2, 0.3],
            meta={"category": "test", "year": 2024},
        )
        result = S3VectorsDocumentStore._document_to_s3_vector(doc)
        assert result["key"] == "test-1"
        assert result["data"] == {"float32": [0.1, 0.2, 0.3]}
        assert result["metadata"]["_content"] == "Hello world"
        assert result["metadata"]["category"] == "test"
        assert result["metadata"]["year"] == 2024

    def test_s3_vector_to_document(self):
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

    def test_roundtrip(self):
        doc = Document(
            id="test-1",
            content="Hello world",
            embedding=[0.1, 0.2, 0.3],
            meta={"category": "test", "year": 2024},
        )
        vector = S3VectorsDocumentStore._document_to_s3_vector(doc)
        restored = S3VectorsDocumentStore._s3_vector_to_document(vector)
        assert restored.id == doc.id
        assert restored.content == doc.content
        assert restored.embedding == doc.embedding
        assert restored.meta == doc.meta
