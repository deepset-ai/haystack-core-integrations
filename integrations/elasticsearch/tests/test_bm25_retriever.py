# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchBM25Retriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert not retriever._scale_score

    retriever = ElasticsearchBM25Retriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        ElasticsearchBM25Retriever(document_store=mock_store, filter_policy="keep")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchBM25Retriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
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
                    "custom_mapping": None,
                    "index": "default",
                    "embedding_similarity_function": "cosine",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": False,
            "filter_policy": "replace",
        },
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": "haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": True,
            "filter_policy": "replace",
        },
    }
    retriever = ElasticsearchBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._fuzziness == "AUTO"
    assert retriever._top_k == 10
    assert retriever._scale_score
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_elasticsearch_client):
    data = {
        "type": "haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": True,
        },
    }
    retriever = ElasticsearchBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._fuzziness == "AUTO"
    assert retriever._top_k == 10
    assert retriever._scale_score
    assert retriever._filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = ElasticsearchBM25Retriever(document_store=mock_store)
    res = retriever.run(query="some query")
    mock_store._bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={},
        fuzziness="AUTO",
        top_k=10,
        scale_score=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._bm25_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchBM25Retriever(document_store=mock_store)

    res = await retriever.run_async(query="some test query")
    mock_store._bm25_retrieval_async.assert_called_once_with(
        query="some test query", filters={}, fuzziness="AUTO", top_k=10, scale_score=False
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


@pytest.mark.asyncio
async def test_run_init_params_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._bm25_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchBM25Retriever(
        document_store=mock_store,
        filters={"some": "filter"},
        fuzziness="3",
        top_k=3,
        scale_score=True,
        filter_policy=FilterPolicy.MERGE,
    )
    res = await retriever.run_async(query="some query")
    mock_store._bm25_retrieval_async.assert_called_once_with(
        query="some query",
        filters={"some": "filter"},
        fuzziness="3",
        top_k=3,
        scale_score=True,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


@pytest.mark.asyncio
async def test_run_time_params_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._bm25_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchBM25Retriever(
        document_store=mock_store,
        filters={"some": "filter"},
        fuzziness="3",
        top_k=3,
        scale_score=True,
        filter_policy=FilterPolicy.MERGE,
    )

    res = await retriever.run_async(query="some query", filters={"another": "filter"}, top_k=1)
    mock_store._bm25_retrieval_async.assert_called_once_with(
        query="some query", filters={"another": "filter"}, top_k=1, fuzziness="3", scale_score=True
    )

    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"
