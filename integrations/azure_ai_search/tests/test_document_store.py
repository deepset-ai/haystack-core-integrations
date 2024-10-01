# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from haystack.dataclasses.document import Document
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    WriteDocumentsTest,
)
from haystack.utils.auth import EnvVarSecret, Secret

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore
from haystack_integrations.document_stores.azure_ai_search.document_store import DEFAULT_VECTOR_SEARCH


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_to_dict(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_SEARCH_SERVICE_ENDPOINT", "test-endpoint")
    document_store = AzureAISearchDocumentStore()
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "azure_endpoint": {"env_vars": ["AZURE_SEARCH_SERVICE_ENDPOINT"], "strict": False, "type": "env_var"},
            "api_key": {"env_vars": ["AZURE_SEARCH_API_KEY"], "strict": False, "type": "env_var"},
            "index_name": "default",
            "embedding_dimension": 768,
            "metadata_fields": None,
            "create_index": True,
            "vector_search_configuration": DEFAULT_VECTOR_SEARCH,
        },
    }


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_from_dict(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_SEARCH_SERVICE_ENDPOINT", "test-endpoint")

    data = {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "azure_endpoint": {"env_vars": ["AZURE_SEARCH_SERVICE_ENDPOINT"], "strict": False, "type": "env_var"},
            "api_key": {"env_vars": ["AZURE_SEARCH_API_KEY"], "strict": False, "type": "env_var"},
            "embedding_dimension": 768,
            "index_name": "default",
            "metadata_fields": None,
            "create_index": False,
            "vector_search_configuration": DEFAULT_VECTOR_SEARCH,
        },
    }
    document_store = AzureAISearchDocumentStore.from_dict(data)
    assert isinstance(document_store._api_key, EnvVarSecret)
    assert isinstance(document_store._azure_endpoint, EnvVarSecret)
    assert document_store._index_name == "default"
    assert document_store._embedding_dimension == 768
    assert document_store._metadata_fields is None
    assert document_store._create_index is False
    assert document_store._vector_search_configuration == DEFAULT_VECTOR_SEARCH


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_init_is_lazy(_mock_azure_search_client):
    AzureAISearchDocumentStore(azure_endppoint=Secret.from_token("test_endpoint"))
    _mock_azure_search_client.assert_not_called()


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_init(_mock_azure_search_client):

    document_store = AzureAISearchDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        azure_endpoint=Secret.from_token("fake_endpoint"),
        index_name="my_index",
        create_index=False,
        embedding_dimension=15,
        metadata_fields={"Title": str, "Pages": int},
    )

    assert document_store._index_name == "my_index"
    assert document_store._create_index is False
    assert document_store._embedding_dimension == 15
    assert document_store._metadata_fields == {"Title": str, "Pages": int}
    assert document_store._vector_search_configuration == DEFAULT_VECTOR_SEARCH


@pytest.mark.skipif(
    not os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", None) and not os.environ.get("AZURE_SEARCH_API_KEY", None),
    reason="Missing AZURE_SEARCH_SERVICE_ENDPOINT or AZURE_SEARCH_API_KEY.",
)
class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):

    @pytest.fixture
    def document_store(self):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        index_name = "haystack_integration_test"
        azure_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
        api_key = os.environ["AZURE_SEARCH_API_KEY"]

        client = SearchIndexClient(azure_endpoint, AzureKeyCredential(api_key))
        if index_name in client.list_index_names():
            client.delete_index(index_name)

        store = AzureAISearchDocumentStore(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            index_name=index_name,
            create_index=True,
            embedding_dimension=15,
        )
        yield store
        client.delete_index(index_name)

    def test_write_documents(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
