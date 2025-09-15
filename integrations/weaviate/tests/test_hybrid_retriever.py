# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.weaviate import WeaviateHybridRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


def test_init_default():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    assert retriever._document_store == mock_document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._alpha is None
    assert retriever._max_vector_distance is None
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = WeaviateHybridRetriever(document_store=mock_document_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        WeaviateHybridRetriever(document_store=mock_document_store, filter_policy="keep_all")


def test_init_with_parameters():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    filters = {"field": "content", "operator": "==", "value": "test"}
    retriever = WeaviateHybridRetriever(
        document_store=mock_document_store,
        filters=filters,
        top_k=5,
        alpha=0.5,
        max_vector_distance=0.8,
        filter_policy=FilterPolicy.MERGE,
    )
    assert retriever._document_store == mock_document_store
    assert retriever._filters == filters
    assert retriever._top_k == 5
    assert retriever._alpha == 0.5
    assert retriever._max_vector_distance == 0.8
    assert retriever._filter_policy == FilterPolicy.MERGE


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_to_dict(_mock_weaviate):
    document_store = WeaviateDocumentStore()
    retriever = WeaviateHybridRetriever(document_store=document_store)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever",
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "alpha": None,
            "max_vector_distance": None,
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
def test_to_dict_with_parameters(_mock_weaviate):
    document_store = WeaviateDocumentStore()
    filters = {"field": "content", "operator": "==", "value": "test"}
    retriever = WeaviateHybridRetriever(
        document_store=document_store,
        filters=filters,
        top_k=5,
        alpha=0.7,
        max_vector_distance=0.6,
        filter_policy=FilterPolicy.MERGE,
    )
    result = retriever.to_dict()
    assert result["init_parameters"]["filters"] == filters
    assert result["init_parameters"]["top_k"] == 5
    assert result["init_parameters"]["alpha"] == 0.7
    assert result["init_parameters"]["max_vector_distance"] == 0.6
    assert result["init_parameters"]["filter_policy"] == "merge"


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_from_dict(_mock_weaviate):
    retriever = WeaviateHybridRetriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever",
            "init_parameters": {
                "filters": {},
                "top_k": 10,
                "alpha": None,
                "max_vector_distance": None,
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
    assert retriever._alpha is None
    assert retriever._max_vector_distance is None


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_from_dict_with_parameters(_mock_weaviate):
    filters = {"field": "content", "operator": "==", "value": "test"}
    retriever = WeaviateHybridRetriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever",
            "init_parameters": {
                "filters": filters,
                "top_k": 5,
                "alpha": 0.8,
                "max_vector_distance": 0.7,
                "filter_policy": "merge",
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
    assert retriever._filters == filters
    assert retriever._top_k == 5
    assert retriever._alpha == 0.8
    assert retriever._max_vector_distance == 0.7
    assert retriever._filter_policy == FilterPolicy.MERGE


def test_run_basic():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Test document", score=0.9)]

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    result = retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3])

    assert "documents" in result
    assert len(result["documents"]) == 1
    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="test query", query_embedding=[0.1, 0.2, 0.3], filters={}, top_k=10, alpha=None, max_vector_distance=None
    )


def test_run_with_runtime_filters():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Filtered document", score=0.8)]

    # Test with REPLACE policy (default)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store, filters={"init": "filter"})
    retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3], filters={"runtime": "filter"})

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={"runtime": "filter"},
        top_k=10,
        alpha=None,
        max_vector_distance=None,
    )


def test_run_with_runtime_parameters():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Runtime params document", score=0.6)]

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3], top_k=5, alpha=0.8, max_vector_distance=0.7)

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="test query", query_embedding=[0.1, 0.2, 0.3], filters={}, top_k=5, alpha=0.8, max_vector_distance=0.7
    )


def test_run_with_init_and_runtime_parameters():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Init and runtime params document", score=0.5)]

    retriever = WeaviateHybridRetriever(
        document_store=mock_document_store, top_k=20, alpha=0.3, max_vector_distance=0.9
    )
    retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3], top_k=15, alpha=0.6, max_vector_distance=0.8)

    # Runtime parameters should override init parameters
    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="test query", query_embedding=[0.1, 0.2, 0.3], filters={}, top_k=15, alpha=0.6, max_vector_distance=0.8
    )


def test_run_empty_query():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = []

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    result = retriever.run(query="", query_embedding=[0.1, 0.2, 0.3])

    assert "documents" in result
    assert len(result["documents"]) == 0
    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="", query_embedding=[0.1, 0.2, 0.3], filters={}, top_k=10, alpha=None, max_vector_distance=None
    )


def test_run_multiple_documents():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_documents = [
        Mock(content="Document 1", score=0.9),
        Mock(content="Document 2", score=0.8),
        Mock(content="Document 3", score=0.7),
    ]
    mock_document_store._hybrid_retrieval.return_value = mock_documents

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    result = retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3], top_k=3)

    assert "documents" in result
    assert len(result["documents"]) == 3
    assert result["documents"] == mock_documents


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_from_dict_no_filter_policy(_mock_weaviate):
    retriever = WeaviateHybridRetriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever",
            "init_parameters": {
                "filters": {},
                "top_k": 10,
                "alpha": None,
                "max_vector_distance": None,
                # filter_policy intentionally omitted
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
    assert retriever._alpha is None
    assert retriever._max_vector_distance is None
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run_with_alpha_zero_runtime():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Doc", score=1.0)]

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    _ = retriever.run(
        query="q",
        query_embedding=[0.1, 0.2],
        alpha=0.0,
    )

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="q",
        query_embedding=[0.1, 0.2],
        filters={},
        top_k=10,
        alpha=0.0,
        max_vector_distance=None,
    )


def test_run_with_alpha_zero_init_and_none_runtime():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Doc", score=1.0)]

    retriever = WeaviateHybridRetriever(document_store=mock_document_store, alpha=0.0)
    _ = retriever.run(
        query="q",
        query_embedding=[0.1, 0.2],
        alpha=None,
    )

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="q",
        query_embedding=[0.1, 0.2],
        filters={},
        top_k=10,
        alpha=0.0,
        max_vector_distance=None,
    )


def test_run_with_max_vector_distance_zero_runtime():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Doc", score=1.0)]

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    _ = retriever.run(
        query="q",
        query_embedding=[0.1, 0.2],
        max_vector_distance=0.0,
    )

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="q",
        query_embedding=[0.1, 0.2],
        filters={},
        top_k=10,
        alpha=None,
        max_vector_distance=0.0,
    )


def test_run_with_max_vector_distance_zero_init_and_none_runtime():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Doc", score=1.0)]

    retriever = WeaviateHybridRetriever(document_store=mock_document_store, max_vector_distance=0.0)
    _ = retriever.run(
        query="q",
        query_embedding=[0.1, 0.2],
        max_vector_distance=None,
    )

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="q",
        query_embedding=[0.1, 0.2],
        filters={},
        top_k=10,
        alpha=None,
        max_vector_distance=0.0,
    )
