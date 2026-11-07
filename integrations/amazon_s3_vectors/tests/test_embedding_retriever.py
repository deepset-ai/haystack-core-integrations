# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.amazon_s3_vectors import AmazonS3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import AmazonS3VectorsDocumentStore


def test_init_default():
    mock_store = Mock(spec=AmazonS3VectorsDocumentStore)
    retriever = AmazonS3VectorsEmbeddingRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE

    retriever = AmazonS3VectorsEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever.filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        AmazonS3VectorsEmbeddingRetriever(document_store=mock_store, filter_policy="invalid")


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_to_dict(_mock_boto3):
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        dimension=768,
        region_name="us-east-1",
        create_bucket_and_index=False,
    )
    retriever = AmazonS3VectorsEmbeddingRetriever(document_store=store, top_k=5)
    d = retriever.to_dict()
    assert d == {
        "type": "haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever.AmazonS3VectorsEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.AmazonS3VectorsDocumentStore",
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
        "type": "haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever.AmazonS3VectorsEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.AmazonS3VectorsDocumentStore",
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
    retriever = AmazonS3VectorsEmbeddingRetriever.from_dict(data)
    assert retriever.top_k == 5
    assert retriever.filter_policy == FilterPolicy.REPLACE
    assert retriever.document_store.vector_bucket_name == "test-bucket"
    assert retriever.document_store.dimension == 768


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_from_dict_no_filter_policy(_mock_boto3):
    """Pipelines serialized with older versions may not have filter_policy."""
    data = {
        "type": "haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever.AmazonS3VectorsEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.AmazonS3VectorsDocumentStore",
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
    retriever = AmazonS3VectorsEmbeddingRetriever.from_dict(data)
    assert retriever.filter_policy == FilterPolicy.REPLACE  # default


def test_run():
    init_filters = {"field": "meta.category", "operator": "==", "value": "news"}
    runtime_filters = {"field": "meta.year", "operator": ">", "value": 2020}
    mock_store = Mock(spec=AmazonS3VectorsDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc")]
    retriever = AmazonS3VectorsEmbeddingRetriever(
        document_store=mock_store,
        filter_policy=FilterPolicy.MERGE,
        filters=init_filters,
    )
    res = retriever.run(query_embedding=[0.5, 0.7], filters=runtime_filters)
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"operator": "AND", "conditions": [init_filters, runtime_filters]},
        top_k=10,
    )
    assert res["documents"] == [Document(content="Test doc")]


def test_close():
    mock_store = Mock(spec=AmazonS3VectorsDocumentStore)
    AmazonS3VectorsEmbeddingRetriever(document_store=mock_store).close()
    mock_store.close.assert_called_once_with()


@pytest.mark.integration
def test_embedding_retrieval(document_store: AmazonS3VectorsDocumentStore) -> None:
    docs = [
        Document(id="r-1", content="First", embedding=[1.0, 0.0] + [0.0] * 766),
        Document(id="r-2", content="Second", embedding=[0.8, 0.6] + [0.0] * 766),
        Document(id="r-3", content="Third", embedding=[0.0, 1.0] + [0.0] * 766),
    ]
    document_store.write_documents(docs)

    results = document_store._embedding_retrieval(query_embedding=[1.0, 0.0] + [0.0] * 766, top_k=2)
    assert [doc.id for doc in results] == ["r-1", "r-2"]
    assert [doc.score for doc in results] == pytest.approx([1.0, 0.8], abs=1e-5)
    assert [doc.content for doc in results] == ["First", "Second"]
    # query_vectors does not return vector data
    assert all(doc.embedding is None for doc in results)


@pytest.mark.integration
def test_embedding_retrieval_with_metadata_filter(document_store: AmazonS3VectorsDocumentStore) -> None:
    docs = [
        Document(id="f-1", content="Sports article", embedding=[0.1] * 768, meta={"category": "sports"}),
        Document(id="f-2", content="News article", embedding=[0.1] * 768, meta={"category": "news"}),
    ]
    document_store.write_documents(docs)

    filters = {"field": "meta.category", "operator": "==", "value": "sports"}
    results = document_store._embedding_retrieval(query_embedding=[0.1] * 768, filters=filters, top_k=10)
    assert len(results) >= 1
    assert all(d.meta.get("category") == "sports" for d in results)


@pytest.mark.integration
def test_retriever_component(document_store: AmazonS3VectorsDocumentStore) -> None:
    document_store.write_documents([Document(id="c-1", content="Hello", embedding=[0.1] * 768)])

    retriever = AmazonS3VectorsEmbeddingRetriever(document_store=document_store, top_k=5)
    result = retriever.run(query_embedding=[0.1] * 768)
    assert "documents" in result
    assert len(result["documents"]) > 0
