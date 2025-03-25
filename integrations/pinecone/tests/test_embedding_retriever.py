# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils import Secret

from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


def test_init_default():
    mock_store = Mock(spec=PineconeDocumentStore)
    retriever = PineconeEmbeddingRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE

    retriever = PineconeEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever.filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        PineconeEmbeddingRetriever(document_store=mock_store, filter_policy="invalid")


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_to_dict(mock_pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 512}
    document_store = PineconeDocumentStore(
        index="default",
        namespace="test-namespace",
        batch_size=50,
        dimension=512,
    )
    retriever = PineconeEmbeddingRetriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "api_key": {
                        "env_vars": [
                            "PINECONE_API_KEY",
                        ],
                        "strict": True,
                        "type": "env_var",
                    },
                    "index": "default",
                    "namespace": "test-namespace",
                    "batch_size": 50,
                    "dimension": 512,
                    "spec": {"serverless": {"region": "us-east-1", "cloud": "aws"}},
                    "metric": "cosine",
                },
                "type": "haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_from_dict(mock_pinecone, monkeypatch):
    data = {
        "type": "haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "api_key": {
                        "env_vars": [
                            "PINECONE_API_KEY",
                        ],
                        "strict": True,
                        "type": "env_var",
                    },
                    "index": "default",
                    "namespace": "test-namespace",
                    "batch_size": 50,
                    "dimension": 512,
                    "spec": {"serverless": {"region": "us-east-1", "cloud": "aws"}},
                    "metric": "cosine",
                },
                "type": "haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }

    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 512}
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    retriever = PineconeEmbeddingRetriever.from_dict(data)

    document_store = retriever.document_store
    assert document_store.api_key == Secret.from_env_var("PINECONE_API_KEY", strict=True)
    assert document_store.index_name == "default"
    assert document_store.namespace == "test-namespace"
    assert document_store.batch_size == 50
    assert document_store.dimension == 512
    assert document_store.metric == "cosine"
    assert document_store.spec == {"serverless": {"region": "us-east-1", "cloud": "aws"}}

    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_from_dict_no_filter_policy(mock_pinecone, monkeypatch):
    data = {
        "type": "haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "api_key": {
                        "env_vars": [
                            "PINECONE_API_KEY",
                        ],
                        "strict": True,
                        "type": "env_var",
                    },
                    "index": "default",
                    "namespace": "test-namespace",
                    "batch_size": 50,
                    "dimension": 512,
                    "spec": {"serverless": {"region": "us-east-1", "cloud": "aws"}},
                    "metric": "cosine",
                },
                "type": "haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }

    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 512}
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    retriever = PineconeEmbeddingRetriever.from_dict(data)

    document_store = retriever.document_store
    assert document_store.api_key == Secret.from_env_var("PINECONE_API_KEY", strict=True)
    assert document_store.index_name == "default"
    assert document_store.namespace == "test-namespace"
    assert document_store.batch_size == 50
    assert document_store.dimension == 512
    assert document_store.metric == "cosine"
    assert document_store.spec == {"serverless": {"region": "us-east-1", "cloud": "aws"}}

    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE


def test_run():
    mock_store = Mock(spec=PineconeDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = PineconeEmbeddingRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=PineconeDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = PineconeEmbeddingRetriever(document_store=mock_store)
    res = await retriever.run_async(query_embedding=[0.5, 0.7])
    assert len(res) == 1
    assert len(res["documents"]) == 1
