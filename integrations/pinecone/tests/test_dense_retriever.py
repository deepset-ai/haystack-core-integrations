# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from pinecone_haystack.dense_retriever import PineconeDenseRetriever
from pinecone_haystack.document_store import PineconeDocumentStore


def test_init_default():
    mock_store = Mock(spec=PineconeDocumentStore)
    retriever = PineconeDenseRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10


@patch("pinecone_haystack.document_store.pinecone")
def test_to_dict(mock_pinecone):
    mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 512}
    document_store = PineconeDocumentStore(
        api_key="test-key",
        environment="gcp-starter",
        index="default",
        namespace="test-namespace",
        batch_size=50,
        dimension=512,
    )
    retriever = PineconeDenseRetriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "pinecone_haystack.dense_retriever.PineconeDenseRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "environment": "gcp-starter",
                    "index": "default",
                    "namespace": "test-namespace",
                    "batch_size": 50,
                    "dimension": 512,
                },
                "type": "pinecone_haystack.document_store.PineconeDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }


@patch("pinecone_haystack.document_store.pinecone")
def test_from_dict(mock_pinecone, monkeypatch):
    data = {
        "type": "pinecone_haystack.dense_retriever.PineconeDenseRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "environment": "gcp-starter",
                    "index": "default",
                    "namespace": "test-namespace",
                    "batch_size": 50,
                    "dimension": 512,
                },
                "type": "pinecone_haystack.document_store.PineconeDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }

    mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 512}
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    retriever = PineconeDenseRetriever.from_dict(data)

    document_store = retriever.document_store
    assert document_store.environment == "gcp-starter"
    assert document_store.index == "default"
    assert document_store.namespace == "test-namespace"
    assert document_store.batch_size == 50
    assert document_store.dimension == 512

    assert retriever.filters == {}
    assert retriever.top_k == 10


def test_run():
    mock_store = Mock(spec=PineconeDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = PineconeDenseRetriever(document_store=mock_store)
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
