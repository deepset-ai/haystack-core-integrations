# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import Mock

import pytest
from azure.core.exceptions import HttpResponseError
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from numpy.random import rand  # type: ignore

from haystack_integrations.components.retrievers.azure_ai_search import AzureAISearchHybridRetriever
from haystack_integrations.document_stores.azure_ai_search import DEFAULT_VECTOR_SEARCH, AzureAISearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    retriever = AzureAISearchHybridRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = AzureAISearchHybridRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        AzureAISearchHybridRetriever(document_store=mock_store, filter_policy="unknown")


def test_to_dict():
    document_store = AzureAISearchDocumentStore(hosts="some fake host")
    retriever = AzureAISearchHybridRetriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "haystack_integrations.components.retrievers.azure_ai_search.hybrid_retriever.AzureAISearchHybridRetriever",  # noqa: E501
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "document_store": {
                "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",  # noqa: E501
                "init_parameters": {
                    "azure_endpoint": {
                        "type": "env_var",
                        "env_vars": ["AZURE_AI_SEARCH_ENDPOINT"],
                        "strict": True,
                    },
                    "api_key": {"type": "env_var", "env_vars": ["AZURE_AI_SEARCH_API_KEY"], "strict": False},
                    "index_name": "default",
                    "embedding_dimension": 768,
                    "metadata_fields": {},
                    "vector_search_configuration": {
                        "profiles": [
                            {"name": "default-vector-config", "algorithm_configuration_name": "cosine-algorithm-config"}
                        ],
                        "algorithms": [
                            {
                                "name": "cosine-algorithm-config",
                                "kind": "hnsw",
                                "parameters": {"m": 4, "ef_construction": 400, "ef_search": 500, "metric": "cosine"},
                            }
                        ],
                    },
                    "hosts": "some fake host",
                },
            },
            "filter_policy": "replace",
        },
    }


def test_from_dict():
    data = {
        "type": "haystack_integrations.components.retrievers.azure_ai_search.hybrid_retriever.AzureAISearchHybridRetriever",  # noqa: E501
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "document_store": {
                "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",  # noqa: E501
                "init_parameters": {
                    "azure_endpoint": {
                        "type": "env_var",
                        "env_vars": ["AZURE_AI_SEARCH_ENDPOINT"],
                        "strict": True,
                    },
                    "api_key": {"type": "env_var", "env_vars": ["AZURE_AI_SEARCH_API_KEY"], "strict": False},
                    "index_name": "default",
                    "embedding_dimension": 768,
                    "metadata_fields": None,
                    "vector_search_configuration": DEFAULT_VECTOR_SEARCH,
                    "hosts": "some fake host",
                },
            },
            "filter_policy": "replace",
        },
    }
    retriever = AzureAISearchHybridRetriever.from_dict(data)
    assert isinstance(retriever._document_store, AzureAISearchDocumentStore)
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    mock_store._hybrid_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = AzureAISearchHybridRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7], query="Test query")
    mock_store._hybrid_retrieval.assert_called_once_with(
        query="Test query",
        query_embedding=[0.5, 0.7],
        filters="",
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_init_params():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    mock_store._hybrid_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = AzureAISearchHybridRetriever(
        document_store=mock_store,
        filters={"field": "type", "operator": "==", "value": "article"},
        top_k=11,
    )
    res = retriever.run(query_embedding=[0.5, 0.7], query="Test query")
    mock_store._hybrid_retrieval.assert_called_once_with(
        query="Test query",
        query_embedding=[0.5, 0.7],
        filters="type eq 'article'",
        top_k=11,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_time_params():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    mock_store._hybrid_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = AzureAISearchHybridRetriever(
        document_store=mock_store,
        filters={"field": "type", "operator": "==", "value": "article"},
        top_k=11,
        select="name",
    )
    res = retriever.run(
        query_embedding=[0.5, 0.7],
        query="Test query",
        filters={"field": "type", "operator": "==", "value": "book"},
        top_k=9,
    )
    mock_store._hybrid_retrieval.assert_called_once_with(
        query="Test query",
        query_embedding=[0.5, 0.7],
        filters="type eq 'book'",
        top_k=9,
        select="name",
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.skipif(
    not os.environ.get("AZURE_AI_SEARCH_ENDPOINT", None) and not os.environ.get("AZURE_AI_SEARCH_API_KEY", None),
    reason="Missing AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_API_KEY.",
)
@pytest.mark.integration
class TestRetriever:
    def test_run(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1")]
        document_store.write_documents(docs)
        retriever = AzureAISearchHybridRetriever(document_store=document_store)
        res = retriever.run(query="Test document", query_embedding=[0.1] * 768)
        assert res["documents"][0].id == docs[0].id

    def test_hybrid_retrieval(self, document_store: AzureAISearchDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 200 + [0.1] * 300 + [0.2] * 268
        another_embedding = rand(768).tolist()

        docs = [
            Document(content="This is first document", embedding=most_similar_embedding),
            Document(content="This is second document", embedding=second_best_embedding),
            Document(content="This is third document", embedding=another_embedding),
        ]

        document_store.write_documents(docs)
        retriever = AzureAISearchHybridRetriever(document_store=document_store)
        results = retriever.run(query="This is first document", query_embedding=query_embedding)
        assert results["documents"][0].content == "This is first document"

    def test_empty_query_embedding(self, document_store: AzureAISearchDocumentStore):
        query_embedding: list[float] = []
        with pytest.raises(ValueError):
            document_store._hybrid_retrieval(query="", query_embedding=query_embedding)

    def test_query_embedding_wrong_dimension(self, document_store: AzureAISearchDocumentStore):
        query_embedding = [0.1] * 4
        with pytest.raises(HttpResponseError):
            document_store._hybrid_retrieval(query="", query_embedding=query_embedding)
