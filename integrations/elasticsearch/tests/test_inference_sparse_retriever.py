# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceSparseRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER")

    assert retriever._document_store == mock_store
    assert retriever._inference_id == "ELSER"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_init_requires_inference_id():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    with pytest.raises(ValueError, match="inference_id must be provided"):
        ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="")


def test_init_wrong_document_store_type():
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchInferenceSparseRetriever(document_store=Mock(), inference_id="ELSER")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host", sparse_vector_field="sparse_vec")
    retriever = ElasticsearchInferenceSparseRetriever(document_store=document_store, inference_id="ELSER")
    retriever_type = (
        "haystack_integrations.components.retrievers.elasticsearch."
        "inference_sparse_retriever.ElasticsearchInferenceSparseRetriever"
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
            "inference_id": "ELSER",
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
            "inference_sparse_retriever.ElasticsearchInferenceSparseRetriever"
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
            "inference_id": "ELSER",
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }
    retriever = ElasticsearchInferenceSparseRetriever.from_dict(data)

    assert retriever._document_store
    assert retriever._inference_id == "ELSER"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "inference_sparse_retriever.ElasticsearchInferenceSparseRetriever"
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
            "inference_id": "ELSER",
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = ElasticsearchInferenceSparseRetriever.from_dict(data)

    assert retriever._document_store
    assert retriever._inference_id == "ELSER"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Test doc")]
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER")
    res = retriever.run(query="Find docs")

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


def test_run_init_params():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Test doc")]
    init_filters = {"field": "meta.source", "operator": "==", "value": "wiki"}
    retriever = ElasticsearchInferenceSparseRetriever(
        document_store=mock_store,
        inference_id="ELSER",
        filters=init_filters,
        top_k=3,
        filter_policy=FilterPolicy.MERGE,
    )

    res = retriever.run(query="Find docs")

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters=init_filters,
        top_k=3,
    )
    assert len(res["documents"]) == 1


def test_run_replace_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceSparseRetriever(
        document_store=mock_store,
        inference_id="ELSER",
        filters={"field": "meta.source", "operator": "==", "value": "wiki"},
        top_k=5,
        filter_policy=FilterPolicy.REPLACE,
    )
    runtime_filters = {"field": "meta.lang", "operator": "==", "value": "en"}

    retriever.run(query="Find docs", filters=runtime_filters)

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters=runtime_filters,
        top_k=5,
    )


def test_run_merge_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = []
    init_filters = {"field": "meta.source", "operator": "==", "value": "wiki"}
    runtime_filters = {"field": "meta.lang", "operator": "==", "value": "en"}
    retriever = ElasticsearchInferenceSparseRetriever(
        document_store=mock_store,
        inference_id="ELSER",
        filters=init_filters,
        top_k=5,
        filter_policy=FilterPolicy.MERGE,
    )

    retriever.run(query="Find docs", filters=runtime_filters)

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={"operator": "AND", "conditions": [init_filters, runtime_filters]},
        top_k=5,
    )


def test_run_runtime_top_k_overrides():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER", top_k=10)

    retriever.run(query="Find docs", top_k=2)

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={},
        top_k=2,
    )


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER")
    res = await retriever.run_async(query="Find docs")

    mock_store._sparse_vector_retrieval_inference_async.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"
