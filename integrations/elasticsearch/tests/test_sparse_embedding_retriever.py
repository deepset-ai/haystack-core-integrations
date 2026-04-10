# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSparseEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10

    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        ElasticsearchSparseEmbeddingRetriever(document_store=mock_store, filter_policy="keep")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host", sparse_vector_field="sparse_vec")
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=document_store)
    retriever_type = (
        "haystack_integrations.components.retrievers.elasticsearch."
        "sparse_embedding_retriever.ElasticsearchSparseEmbeddingRetriever"
    )
    assert retriever.to_dict() == {
        "type": retriever_type,
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
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "sparse_embedding_retriever.ElasticsearchSparseEmbeddingRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }
    retriever = ElasticsearchSparseEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "sparse_embedding_retriever.ElasticsearchSparseEmbeddingRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = ElasticsearchSparseEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval.return_value = [Document(content="Test doc")]
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store)
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = retriever.run(query_sparse_embedding=query_sparse_embedding)
    mock_store._sparse_vector_retrieval.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store)
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = await retriever.run_async(query_sparse_embedding=query_sparse_embedding)
    mock_store._sparse_vector_retrieval_async.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


@pytest.mark.asyncio
async def test_run_init_params_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchSparseEmbeddingRetriever(
        document_store=mock_store,
        filters={"some": "filter"},
        top_k=3,
        filter_policy=FilterPolicy.MERGE,
    )
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = await retriever.run_async(query_sparse_embedding=query_sparse_embedding)
    mock_store._sparse_vector_retrieval_async.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters={"some": "filter"},
        top_k=3,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


@pytest.mark.asyncio
async def test_run_time_params_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchSparseEmbeddingRetriever(
        document_store=mock_store,
        filters={"some": "filter"},
        top_k=3,
        filter_policy=FilterPolicy.MERGE,
    )
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = await retriever.run_async(
        query_sparse_embedding=query_sparse_embedding,
        filters={"another": "filter"},
        top_k=1,
    )
    mock_store._sparse_vector_retrieval_async.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters={"another": "filter"},
        top_k=1,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"
