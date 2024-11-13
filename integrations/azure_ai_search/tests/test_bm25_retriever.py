# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List
from unittest.mock import Mock

import pytest
from azure.core.exceptions import HttpResponseError
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from numpy.random import rand  # type: ignore

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
        "type": "haystack_integrations.components.retrievers.azure_ai_search.bm25_retriever.AzureAISearchBM25Retriever",  # noqa: E501
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
                    "embedding_dimension": 768,
                    "metadata_fields": None,
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
        "type": "haystack_integrations.components.retrievers.azure_ai_search.bm25_retriever.AzureAISearchBM25Retriever",  # noqa: E501
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


@pytest.mark.skipif(
    not os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", None) and not os.environ.get("AZURE_SEARCH_API_KEY", None),
    reason="Missing AZURE_SEARCH_SERVICE_ENDPOINT or AZURE_SEARCH_API_KEY.",
)
@pytest.mark.integration
class TestRetriever:

    def test_run(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1", content="Test document")]
        document_store.write_documents(docs)
        retriever = AzureAISearchBM25Retriever(document_store=document_store)
        res = retriever.run(query="Test document")
        assert res["documents"] == docs

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
