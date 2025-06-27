# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.azure_ai_search import AzureAISearchBM25Retriever
from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    retriever = AzureAISearchBM25Retriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = AzureAISearchBM25Retriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        AzureAISearchBM25Retriever(document_store=mock_store, filter_policy="unknown")


def test_to_dict():
    document_store = AzureAISearchDocumentStore(hosts="some fake host")
    retriever = AzureAISearchBM25Retriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "haystack_integrations.components.retrievers.azure_ai_search.bm25_retriever.AzureAISearchBM25Retriever",
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
        "type": "haystack_integrations.components.retrievers.azure_ai_search.bm25_retriever.AzureAISearchBM25Retriever",
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "document_store": {
                "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",  # noqa: E501
                "init_parameters": {
                    "azure_endpoint": {
                        "type": "env_var",
                        "env_vars": ["AZURE_SEARCH_SERVICE_ENDPOINT"],
                        "strict": True,
                    },
                    "api_key": {"type": "env_var", "env_vars": ["AZURE_SEARCH_API_KEY"], "strict": False},
                    "index_name": "default",
                    "metadata_fields": None,
                    "hosts": "some fake host",
                },
            },
            "filter_policy": "replace",
        },
    }
    retriever = AzureAISearchBM25Retriever.from_dict(data)
    assert isinstance(retriever._document_store, AzureAISearchDocumentStore)
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = AzureAISearchBM25Retriever(document_store=mock_store)
    res = retriever.run(query="Test query")
    mock_store._bm25_retrieval.assert_called_once_with(
        query="Test query",
        filters="",
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


def test_run_init_params():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = AzureAISearchBM25Retriever(
        document_store=mock_store, filters={"field": "type", "operator": "==", "value": "article"}, top_k=11
    )
    res = retriever.run(query="Test query")
    mock_store._bm25_retrieval.assert_called_once_with(
        query="Test query",
        filters="type eq 'article'",
        top_k=11,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


def test_run_time_params():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = AzureAISearchBM25Retriever(
        document_store=mock_store,
        filters={"field": "type", "operator": "==", "value": "article"},
        top_k=11,
        select="name",
    )
    res = retriever.run(query="Test query", filters={"field": "type", "operator": "==", "value": "book"}, top_k=5)
    mock_store._bm25_retrieval.assert_called_once_with(
        query="Test query", filters="type eq 'book'", top_k=5, select="name"
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


@pytest.mark.skipif(
    not os.environ.get("AZURE_AI_SEARCH_ENDPOINT", None) and not os.environ.get("AZURE_AI_SEARCH_API_KEY", None),
    reason="Missing AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_API_KEY.",
)
@pytest.mark.integration
class TestRetriever:
    def test_run(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1", content="Test document")]
        document_store.write_documents(docs)
        retriever = AzureAISearchBM25Retriever(document_store=document_store)
        res = retriever.run(query="Test document")
        assert res["documents"][0].content == docs[0].content
        assert res["documents"][0].score is not None
        assert res["documents"][0].id == docs[0].id

    @pytest.mark.parametrize(
        "document_store",
        [
            {"include_search_metadata": True},
        ],
        indirect=True,
    )
    def test_run_with_search_metadata(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1", content="Test document")]
        document_store.write_documents(docs)
        retriever = AzureAISearchBM25Retriever(document_store=document_store)
        res = retriever.run(query="Test document")
        assert all(key.startswith("@search") for key in res["documents"][0].meta.keys())

    def test_document_retrieval(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(content="This is first document"),
            Document(content="This is second document"),
            Document(content="This is third document"),
        ]

        document_store.write_documents(docs)
        retriever = AzureAISearchBM25Retriever(document_store=document_store)
        results = retriever.run(query="This is first document")
        assert results["documents"][0].content == "This is first document"
