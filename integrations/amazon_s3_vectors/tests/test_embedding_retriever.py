# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


def test_init_default():
    mock_store = Mock(spec=S3VectorsDocumentStore)
    retriever = S3VectorsEmbeddingRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE

    retriever = S3VectorsEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever.filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        S3VectorsEmbeddingRetriever(document_store=mock_store, filter_policy="invalid")


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_to_dict(_mock_boto3):
    store = S3VectorsDocumentStore(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        dimension=768,
        region_name="us-east-1",
        create_bucket_and_index=False,
    )
    retriever = S3VectorsEmbeddingRetriever(document_store=store, top_k=5)
    d = retriever.to_dict()
    assert d == {
        "type": "haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever.S3VectorsEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
                "init_parameters": {
                    "vector_bucket_name": "test-bucket",
                    "index_name": "test-index",
                    "dimension": 768,
                    "distance_metric": "cosine",
                    "region_name": "us-east-1",
                    "aws_access_key_id": None,
                    "aws_secret_access_key": None,
                    "aws_session_token": None,
                    "create_bucket_and_index": False,
                    "non_filterable_metadata_keys": [],
                },
            },
            "filters": {},
            "top_k": 5,
            "filter_policy": "replace",
        },
    }


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_from_dict(_mock_boto3):
    data = {
        "type": "haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever.S3VectorsEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
                "init_parameters": {
                    "vector_bucket_name": "test-bucket",
                    "index_name": "test-index",
                    "dimension": 768,
                    "distance_metric": "cosine",
                    "region_name": "us-east-1",
                    "aws_access_key_id": None,
                    "aws_secret_access_key": None,
                    "aws_session_token": None,
                    "create_bucket_and_index": False,
                    "non_filterable_metadata_keys": [],
                },
            },
            "filters": {},
            "top_k": 5,
            "filter_policy": "replace",
        },
    }
    retriever = S3VectorsEmbeddingRetriever.from_dict(data)
    assert retriever.top_k == 5
    assert retriever.filter_policy == FilterPolicy.REPLACE
    assert retriever.document_store.vector_bucket_name == "test-bucket"
    assert retriever.document_store.dimension == 768


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_from_dict_no_filter_policy(_mock_boto3):
    """Pipelines serialized with older versions may not have filter_policy."""
    data = {
        "type": "haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever.S3VectorsEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
                "init_parameters": {
                    "vector_bucket_name": "test-bucket",
                    "index_name": "test-index",
                    "dimension": 768,
                    "create_bucket_and_index": False,
                },
            },
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = S3VectorsEmbeddingRetriever.from_dict(data)
    assert retriever.filter_policy == FilterPolicy.REPLACE  # default


def test_run():
    mock_store = Mock(spec=S3VectorsDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = S3VectorsEmbeddingRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
    )
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
