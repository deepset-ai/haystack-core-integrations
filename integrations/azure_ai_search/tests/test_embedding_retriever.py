# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List
from unittest.mock import Mock

import pytest
from azure.core.exceptions import HttpResponseError
from haystack.dataclasses import Document, Pipeline
from haystack.document_stores.types import FilterPolicy
from numpy.random import rand

from haystack_integrations.components.retrievers.azure_ai_search import AzureAISearchEmbeddingRetriever
from haystack_integrations.document_stores.azure_ai_search import DEFAULT_VECTOR_SEARCH, AzureAISearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=AzureAISearchDocumentStore)
    retriever = AzureAISearchEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = AzureAISearchEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        AzureAISearchEmbeddingRetriever(document_store=mock_store, filter_policy="unknown")


def test_to_dict():
    document_store = AzureAISearchDocumentStore(hosts="some fake host")
    retriever = AzureAISearchEmbeddingRetriever(document_store=document_store)
    res = retriever.to_dict()
    type_s = "haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever"
    assert res == {
        "type": type_s,
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "document_store": {
                "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
                "init_parameters": {
                    "azure_endpoint": {
                        "type": "env_var",
                        "env_vars": ["AZURE_SEARCH_SERVICE_ENDPOINT"],
                        "strict": False,
                    },
                    "api_key": {"type": "env_var", "env_vars": ["AZURE_SEARCH_API_KEY"], "strict": False},
                    "index_name": "default",
                    "create_index": True,
                    "embedding_dimension": 768,
                    "metadata_fields": None,
                    "vector_search_configuration": DEFAULT_VECTOR_SEARCH,
                    "hosts": "some fake host",
                },
            },
            "filter_policy": "replace",
        },
    }


def test_from_dict():
    type_s = "haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever"
    data = {
        "type": type_s,
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "document_store": {
                "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
                "init_parameters": {
                    "azure_endpoint": {
                        "type": "env_var",
                        "env_vars": ["AZURE_SEARCH_SERVICE_ENDPOINT"],
                        "strict": False,
                    },
                    "api_key": {"type": "env_var", "env_vars": ["AZURE_SEARCH_API_KEY"], "strict": False},
                    "index_name": "default",
                    "create_index": True,
                    "embedding_dimension": 768,
                    "metadata_fields": None,
                    "vector_search_configuration": DEFAULT_VECTOR_SEARCH,
                    "hosts": "some fake host",
                },
            },
            "filter_policy": "replace",
        },
    }
    retriever = AzureAISearchEmbeddingRetriever.from_dict(data)
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
        docs = [Document(id="1")]
        document_store.write_documents(docs)
        retriever = AzureAISearchEmbeddingRetriever(document_store=document_store)
        res = retriever.run(query_embedding=[0.1] * 15)
        assert res["documents"] == docs

    def test_embedding_retrieval(self, document_store: AzureAISearchDocumentStore):
        query_embedding = [0.1] * 15
        most_similar_embedding = [0.8] * 15
        second_best_embedding = [0.8] * 7 + [0.1] * 3 + [0.2] * 5
        another_embedding = rand(15).tolist()

        docs = [
            Document(content="This is first document", embedding=most_similar_embedding),
            Document(content="This is second document", embedding=second_best_embedding),
            Document(content="This is thrid document", embedding=another_embedding),
        ]

        document_store.write_documents(docs)
        retriever = AzureAISearchEmbeddingRetriever(document_store=document_store)
        results = retriever.run(query_embedding=query_embedding)
        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=1)
        assert len(results) == 1
        assert results[0].content == "This is first document"

    def test_empty_query_embedding(self, document_store: AzureAISearchDocumentStore):
        query_embedding: List[float] = []
        with pytest.raises(ValueError):
            document_store._embedding_retrieval(query_embedding=query_embedding)

    def test_query_embedding_wrong_dimension(self, document_store: AzureAISearchDocumentStore):
        query_embedding = [0.1] * 4
        with pytest.raises(HttpResponseError):
            document_store._embedding_retrieval(query_embedding=query_embedding)
