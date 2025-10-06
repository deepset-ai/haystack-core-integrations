# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates is None

    retriever = ElasticsearchEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        ElasticsearchEmbeddingRetriever(document_store=mock_store, filter_policy="keep")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)
    res = retriever.to_dict()
    t = "haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever"
    assert res == {
        "type": t,
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "api_key": {
                        "env_vars": [
                            "ELASTIC_API_KEY",
                        ],
                        "strict": False,
                        "type": "env_var",
                    },
                    "api_key_id": {
                        "env_vars": [
                            "ELASTIC_API_KEY_ID",
                        ],
                        "strict": False,
                        "type": "env_var",
                    },
                    "hosts": "some fake host",
                    "custom_mapping": None,
                    "index": "default",
                    "embedding_similarity_function": "cosine",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "num_candidates": None,
        },
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    t = "haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever"
    data = {
        "type": t,
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "num_candidates": None,
        },
    }
    retriever = ElasticsearchEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates is None


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_elasticsearch_client):
    t = "haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever"
    data = {
        "type": t,
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
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
    assert retriever._filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE


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


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="test document", embedding=[0.1, 0.2])]
    retriever = ElasticsearchEmbeddingRetriever(document_store=mock_store)
    res = await retriever.run_async(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        num_candidates=None,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_init_params_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="test document", embedding=[0.1, 0.2])]
    retriever = ElasticsearchEmbeddingRetriever(
        document_store=mock_store,
        filters={"some": "filter"},
        top_k=3,
        num_candidates=30,
        filter_policy=FilterPolicy.MERGE,
    )
    res = await retriever.run_async(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"some": "filter"},
        top_k=3,
        num_candidates=30,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_time_params_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="test document", embedding=[0.1, 0.2])]
    retriever = ElasticsearchEmbeddingRetriever(
        document_store=mock_store,
        filters={"some": "filter"},
        top_k=3,
        num_candidates=30,
        filter_policy=FilterPolicy.MERGE,
    )

    res = await retriever.run_async(query_embedding=[0.5, 0.7], filters={"another": "filter"}, top_k=1)
    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7], filters={"another": "filter"}, top_k=1, num_candidates=30
    )

    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"
    assert res["documents"][0].embedding == [0.1, 0.2]
