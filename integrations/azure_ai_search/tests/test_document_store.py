# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import random
from typing import List
from unittest.mock import patch

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore
from haystack_integrations.document_stores.azure_ai_search.document_store import MAX_UPLOAD_BATCH_SIZE
from azure.core.exceptions import ResourceNotFoundError


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_to_dict(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    document_store = AzureAISearchDocumentStore(azure_endpoint="some endpoint")
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
            "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
            "embedding_dimensions": 768,
            "azure_endpoint": "some endpoint",
            "index_name": "default",
            "metadata_fields": None,
            "create_index": True,
        },
    }


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_from_dict(_mock_azure_ai_search_client):
    data = {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "embedding_dimesnions": 768,
            "azure_endpoint": "some endpoint",
            "index_name": "default",
            "metadata_fields": None,
            "create_index": True,
        },
    }
    document_store = AzureAISearchDocumentStore.from_dict(data)
    assert document_store._azure_endpoint == "some endpoint"
    assert document_store._index_name == "default"
    assert document_store._embedding_dimensions == 768
    assert document_store._metadata_fields is None
    assert document_store._create_index is False


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearch")
def test_init_is_lazy(_mock_azure_search_client):
    AzureAISearchDocumentStore(azure_endppoint="test_endpoint")
    _mock_azure_search_client.assert_not_called()



