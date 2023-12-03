# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from opensearch_haystack.document_store import OpenSearchDocumentStore
from opensearch_haystack.embedding_retriever import OpenSearchEmbeddingRetriever


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


@patch("opensearch_haystack.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchEmbeddingRetriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "opensearch_haystack.embedding_retriever.OpenSearchEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                },
                "type": "opensearch_haystack.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }


@patch("opensearch_haystack.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    data = {
        "type": "opensearch_haystack.embedding_retriever.OpenSearchEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "opensearch_haystack.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = OpenSearchEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


def test_run():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
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


def test_run_init_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, filters={"from": "init"}, top_k=11)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "init"},
        top_k=11,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_time_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, filters={"from": "init"}, top_k=11)
    res = retriever.run(query_embedding=[0.5, 0.7], filters={"from": "run"}, top_k=9)
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "run"},
        top_k=9,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]
