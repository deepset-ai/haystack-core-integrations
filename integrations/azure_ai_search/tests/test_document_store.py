# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch
from typing import List
import random

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    WriteDocumentsTest,
    FilterDocumentsTest
)
from haystack.utils.auth import EnvVarSecret, Secret

from haystack_integrations.document_stores.azure_ai_search import DEFAULT_VECTOR_SEARCH, AzureAISearchDocumentStore


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

    def test_write_documents(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    # Parametrize the test with metadata fields
    @pytest.mark.parametrize(
        "document_store",
        [
            {"metadata_fields": {"author": str, "publication_year": int, "rating": float}},
        ],
        indirect=True,
    )
    def test_write_documents_with_meta(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(
                id="1",
                meta={"author": "Tom", "publication_year": 2021, "rating": 4.5},
                content="This is a test document.",
            )
        ]
        document_store.write_documents(docs)
        doc = document_store.get_documents_by_id(["1"])
        assert doc[0] == docs[0]

def _random_embeddings(n):
        return [random.random() for _ in range(n)]
TEST_EMBEDDING_1 = _random_embeddings(768)
TEST_EMBEDDING_2 = _random_embeddings(768)

@pytest.mark.skipif(
    not os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", None) and not os.environ.get("AZURE_SEARCH_API_KEY", None),
    reason="Missing AZURE_SEARCH_SERVICE_ENDPOINT or AZURE_SEARCH_API_KEY.",
)
@pytest.mark.parametrize(
        "document_store",
        [
            {"metadata_fields": {"name": str, "page": str, "chapter": str, "number": int, "date": str}},
        ],
        indirect=True,
    )
class TestFilters(FilterDocumentsTest):
    
    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        """Fixture that returns a list of Documents that can be used to test filtering."""
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "100",
                        "chapter": "intro",
                        "number": 2,
                        "date": "1969-07-21T20:17:40",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "123",
                        "chapter": "abstract",
                        "number": -2,
                        "date": "1972-12-11T19:54:58",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "90",
                        "chapter": "conclusion",
                        "number": -10,
                        "date": "1989-11-09T17:53:00",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            
            documents.append(
                Document(content=f"Doc {i} with zeros emb", meta={"name": "zeros_doc"}, embedding=TEST_EMBEDDING_1)
            )
            documents.append(
                Document(content=f"Doc {i} with ones emb", meta={"name": "ones_doc"}, embedding=TEST_EMBEDDING_2)
            )
        return documents

    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs):
        pass

    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with == comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": None})
        print (result)
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") is None])
