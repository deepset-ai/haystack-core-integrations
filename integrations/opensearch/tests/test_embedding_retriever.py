# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.document_store import DEFAULT_MAX_CHUNK_BYTES


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        OpenSearchEmbeddingRetriever(document_store=mock_store, filter_policy="unknown")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchEmbeddingRetriever(document_store=document_store, custom_query={"some": "custom query"})
    res = retriever.to_dict()
    type_s = "haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever"
    assert res == {
        "type": type_s,
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "embedding_dim": 768,
                    "hosts": "some fake host",
                    "index": "default",
                    "mappings": {
                        "dynamic_templates": [
                            {
                                "strings": {
                                    "mapping": {
                                        "type": "keyword",
                                    },
                                    "match_mapping_type": "string",
                                },
                            },
                        ],
                        "properties": {
                            "content": {
                                "type": "text",
                            },
                            "embedding": {
                                "dimension": 768,
                                "index": True,
                                "type": "knn_vector",
                            },
                        },
                    },
                    "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                    "method": None,
                    "settings": {
                        "index.knn": True,
                    },
                    "return_embedding": False,
                    "create_index": True,
                    "http_auth": None,
                    "use_ssl": None,
                    "verify_certs": None,
                    "timeout": None,
                },
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": True,
        },
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    type_s = "haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever"
    data = {
        "type": type_s,
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": False,
        },
    }
    retriever = OpenSearchEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._custom_query == {"some": "custom query"}
    assert retriever._raise_on_failure is False
    assert retriever._filter_policy == FilterPolicy.REPLACE

    # For backwards compatibility with older versions of the retriever without a filter policy
    data = {
        "type": type_s,
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "custom_query": {"some": "custom query"},
            "raise_on_failure": False,
        },
    }
    retriever = OpenSearchEmbeddingRetriever.from_dict(data)
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        custom_query=None,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_init_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(
        document_store=mock_store, filters={"from": "init"}, top_k=11, custom_query="custom_query"
    )
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "init"},
        top_k=11,
        custom_query="custom_query",
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
        custom_query=None,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_ignore_errors(caplog):
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.side_effect = Exception("Some error")
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, raise_on_failure=False)
    res = retriever.run(query_embedding=[0.5, 0.7])
    assert len(res) == 1
    assert res["documents"] == []
    assert "Some error" in caplog.text
