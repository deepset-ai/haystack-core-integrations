# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from elasticsearch_haystack.document_store import ElasticsearchDocumentStore
from elasticsearch_haystack.embedding_retriever import ElasticsearchEmbeddingRetriever


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates is None


@patch("elasticsearch_haystack.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "elasticsearch_haystack.embedding_retriever.ElasticsearchEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "embedding_similarity_function": "cosine",
                },
                "type": "elasticsearch_haystack.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "num_candidates": None,
        },
    }


@patch("elasticsearch_haystack.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": "elasticsearch_haystack.embedding_retriever.ElasticsearchEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "elasticsearch_haystack.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "num_candidates": None,
        },
    }
    retriever = ElasticsearchEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates is None


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = ElasticsearchEmbeddingRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        num_candidates=None,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]
