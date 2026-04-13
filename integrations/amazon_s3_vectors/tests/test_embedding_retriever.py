# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
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
        store._get_client()
        return store


class TestInit:
    def test_valid_init(self, doc_store):
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store)
        assert retriever.document_store == doc_store
        assert retriever.top_k == 10
        assert retriever.filters == {}
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_invalid_store_raises(self):
        with pytest.raises(ValueError, match="must be an instance"):
            S3VectorsEmbeddingRetriever(document_store="not a store")

    def test_custom_params(self, doc_store):
        retriever = S3VectorsEmbeddingRetriever(
            document_store=doc_store,
            top_k=5,
            filters={"operator": "AND", "conditions": []},
            filter_policy=FilterPolicy.MERGE,
        )
        assert retriever.top_k == 5
        assert retriever.filter_policy == FilterPolicy.MERGE


class TestSerialization:
    def test_to_dict(self, doc_store):
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store, top_k=5)
        d = retriever.to_dict()
        assert d["init_parameters"]["top_k"] == 5
        assert "document_store" in d["init_parameters"]

    def test_from_dict(self, doc_store):
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store, top_k=5)
        d = retriever.to_dict()
        with patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3"):
            restored = S3VectorsEmbeddingRetriever.from_dict(d)
        assert restored.top_k == 5


class TestRun:
    def test_run(self, doc_store, mock_client):
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
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store)
        result = retriever.run(query_embedding=[0.1] * 768)
        assert len(result["documents"]) == 1
        assert result["documents"][0].id == "1"
        # cosine distance 0.05 -> similarity 0.95
        assert result["documents"][0].score == pytest.approx(0.95)

    def test_run_with_top_k_override(self, doc_store, mock_client):
        mock_client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store, top_k=10)
        retriever.run(query_embedding=[0.1] * 768, top_k=3)
        call_args = mock_client.query_vectors.call_args[1]
        assert call_args["topK"] == 3

    def test_run_with_filters(self, doc_store, mock_client):
        mock_client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
        retriever = S3VectorsEmbeddingRetriever(document_store=doc_store)
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
            ],
        }
        retriever.run(query_embedding=[0.1] * 768, filters=filters)
        call_args = mock_client.query_vectors.call_args[1]
        assert "filter" in call_args
