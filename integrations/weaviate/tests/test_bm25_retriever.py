# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.document_stores.types import FilterPolicy
from haystack_integrations.components.retrievers.weaviate import WeaviateBM25Retriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


def test_init_default():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    retriever = WeaviateBM25Retriever(document_store=mock_document_store)
    assert retriever._document_store == mock_document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = WeaviateBM25Retriever(document_store=mock_document_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        WeaviateBM25Retriever(document_store=mock_document_store, filter_policy="keep_all")


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_to_dict(_mock_weaviate):
    document_store = WeaviateDocumentStore()
    retriever = WeaviateBM25Retriever(document_store=document_store)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever",
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "document_store": {
                "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                "init_parameters": {
                    "url": None,
                    "collection_settings": {
                        "class": "Default",
                        "invertedIndexConfig": {"indexNullState": True},
                        "properties": [
                            {"name": "_original_id", "dataType": ["text"]},
                            {"name": "content", "dataType": ["text"]},
                            {"name": "dataframe", "dataType": ["text"]},
                            {"name": "blob_data", "dataType": ["blob"]},
                            {"name": "blob_mime_type", "dataType": ["text"]},
                            {"name": "score", "dataType": ["number"]},
                        ],
                    },
                    "auth_client_secret": None,
                    "additional_headers": None,
                    "embedded_options": None,
                    "additional_config": None,
                },
            },
        },
    }


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_from_dict(_mock_weaviate):
    retriever = WeaviateBM25Retriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever",
            "init_parameters": {
                "filters": {},
                "top_k": 10,
                "filter_policy": "replace",
                "document_store": {
                    "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                    "init_parameters": {
                        "url": None,
                        "collection_settings": {
                            "class": "Default",
                            "invertedIndexConfig": {"indexNullState": True},
                            "properties": [
                                {"name": "_original_id", "dataType": ["text"]},
                                {"name": "content", "dataType": ["text"]},
                                {"name": "dataframe", "dataType": ["text"]},
                                {"name": "blob_data", "dataType": ["blob"]},
                                {"name": "blob_mime_type", "dataType": ["text"]},
                                {"name": "score", "dataType": ["number"]},
                            ],
                        },
                        "auth_client_secret": None,
                        "additional_headers": None,
                        "embedded_options": None,
                        "additional_config": None,
                    },
                },
            },
        }
    )
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_from_dict_no_filter_policy(_mock_weaviate):
    retriever = WeaviateBM25Retriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever",
            "init_parameters": {
                "filters": {},
                "top_k": 10,
                "document_store": {
                    "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                    "init_parameters": {
                        "url": None,
                        "collection_settings": {
                            "class": "Default",
                            "invertedIndexConfig": {"indexNullState": True},
                            "properties": [
                                {"name": "_original_id", "dataType": ["text"]},
                                {"name": "content", "dataType": ["text"]},
                                {"name": "dataframe", "dataType": ["text"]},
                                {"name": "blob_data", "dataType": ["blob"]},
                                {"name": "blob_mime_type", "dataType": ["text"]},
                                {"name": "score", "dataType": ["number"]},
                            ],
                        },
                        "auth_client_secret": None,
                        "additional_headers": None,
                        "embedded_options": None,
                        "additional_config": None,
                    },
                },
            },
        }
    )
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateDocumentStore")
def test_run(mock_document_store):
    retriever = WeaviateBM25Retriever(document_store=mock_document_store)
    query = "some query"
    filters = {"field": "content", "operator": "==", "value": "Some text"}
    retriever.run(query=query, filters=filters, top_k=5)
    mock_document_store._bm25_retrieval.assert_called_once_with(query=query, filters=filters, top_k=5)
