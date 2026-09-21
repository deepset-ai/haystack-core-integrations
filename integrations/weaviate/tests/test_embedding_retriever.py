# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import create_autospec, patch

import pytest
from haystack import Pipeline
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


def test_close():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store)

    retriever.close()

    mock_document_store.close.assert_called_once_with()
    assert retriever._document_store is mock_document_store


def test_init_default():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store)
    assert retriever._document_store == mock_document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert retriever._distance is None
    assert retriever._certainty is None

    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        WeaviateEmbeddingRetriever(document_store=mock_document_store, filter_policy="keep_all")


def test_init_with_distance_and_certainty():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    with pytest.raises(ValueError, match=r"Can't use 'distance' \(0.1\) and 'certainty' \(0.8\) parameters together"):
        WeaviateEmbeddingRetriever(document_store=mock_document_store, distance=0.1, certainty=0.8)


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_to_dict(_mock_weaviate):
    document_store = WeaviateDocumentStore()
    retriever = WeaviateEmbeddingRetriever(document_store=document_store)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever",
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "distance": None,
            "certainty": None,
            "document_store": {
                "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                "init_parameters": {
                    "url": None,
                    "grpc_port": 50051,
                    "grpc_secure": False,
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
def test_from_dict(_mock_weaviate):
    retriever = WeaviateEmbeddingRetriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever",  # noqa: E501
            "init_parameters": {
                "filters": {},
                "top_k": 10,
                "filter_policy": "replace",
                "distance": None,
                "certainty": None,
                "document_store": {
                    "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                    "init_parameters": {
                        "url": None,
                        "grpc_port": 50051,
                        "grpc_secure": False,
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
    assert retriever._distance is None
    assert retriever._certainty is None


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
def test_from_dict_no_filter_policy(_mock_weaviate):
    retriever = WeaviateEmbeddingRetriever.from_dict(
        {
            "type": "haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever",  # noqa: E501
            "init_parameters": {
                "filters": {},
                "top_k": 10,
                "distance": None,
                "certainty": None,
                "document_store": {
                    "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                    "init_parameters": {
                        "url": None,
                        "grpc_port": 50051,
                        "grpc_secure": False,
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
    assert retriever._distance is None
    assert retriever._certainty is None
    assert retriever._filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE


def test_run():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store)
    query_embedding = [0.1, 0.1, 0.1, 0.1]
    filters = {"field": "content", "operator": "==", "value": "Some text"}
    retriever.run(query_embedding=query_embedding, filters=filters, top_k=5, distance=0.1)
    mock_document_store._embedding_retrieval.assert_called_once_with(
        query_embedding=query_embedding, filters=filters, top_k=5, distance=0.1, certainty=None
    )


@pytest.mark.parametrize("param", ["distance", "certainty"])
def test_run_honors_explicit_zero(param):
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store, **{param: 0.5})

    retriever.run(query_embedding=[0.1, 0.2, 0.3], **{param: 0.0})

    assert mock_document_store._embedding_retrieval.call_args.kwargs[param] == 0.0


@pytest.mark.parametrize("param", ["distance", "certainty"])
def test_run_honors_zero_set_at_init(param):
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store, **{param: 0.0})

    retriever.run(query_embedding=[0.1, 0.2, 0.3])

    assert mock_document_store._embedding_retrieval.call_args.kwargs[param] == 0.0


def test_run_with_distance_and_certainty():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store)
    with pytest.raises(ValueError, match=r"Can't use 'distance' \(0.5\) and 'certainty' \(0.8\) parameters together"):
        retriever.run(query_embedding=[0.1, 0.2, 0.3], distance=0.5, certainty=0.8)


def test_pipeline_serde():
    """A pipeline holding the retriever must survive a dumps/loads round trip, nested store included."""
    document_store = WeaviateDocumentStore(url="http://localhost:8080")
    pipeline = Pipeline()
    pipeline.add_component(
        "retriever", WeaviateEmbeddingRetriever(document_store=document_store, top_k=3, distance=0.5)
    )

    reloaded = Pipeline.loads(pipeline.dumps()).get_component("retriever")

    assert isinstance(reloaded, WeaviateEmbeddingRetriever)
    assert isinstance(reloaded._document_store, WeaviateDocumentStore)
    assert reloaded._document_store.to_dict() == document_store.to_dict()
    assert reloaded._top_k == 3
    assert reloaded._distance == 0.5
