# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore


@patch.dict(
    "os.environ",
    {"ASTRA_DB_APPLICATION_TOKEN": "fake-token", "ASTRA_DB_API_ENDPOINT": "http://fake-url.apps.astra.datastax.com"},
)
@patch("haystack_integrations.document_stores.astra.document_store.AstraClient")
def test_retriever_to_json(*_):
    ds = AstraDocumentStore()

    retriever = AstraEmbeddingRetriever(ds, filters={"foo": "bar"}, top_k=99)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever",
        "init_parameters": {
            "filters": {"foo": "bar"},
            "top_k": 99,
            "document_store": {
                "type": "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore",
                "init_parameters": {
                    "api_endpoint": {"type": "env_var", "env_vars": ["ASTRA_DB_API_ENDPOINT"], "strict": True},
                    "token": {"type": "env_var", "env_vars": ["ASTRA_DB_APPLICATION_TOKEN"], "strict": True},
                    "collection_name": "documents",
                    "embedding_dimension": 768,
                    "duplicates_policy": "NONE",
                    "similarity": "cosine",
                },
            },
        },
    }


@patch.dict(
    "os.environ",
    {"ASTRA_DB_APPLICATION_TOKEN": "fake-token", "ASTRA_DB_API_ENDPOINT": "http://fake-url.apps.astra.datastax.com"},
)
@patch("haystack_integrations.document_stores.astra.document_store.AstraClient")
def test_retriever_from_json(*_):

    data = {
        "type": "haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever",
        "init_parameters": {
            "filters": {"bar": "baz"},
            "top_k": 42,
            "document_store": {
                "type": "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore",
                "init_parameters": {
                    "api_endpoint": {"type": "env_var", "env_vars": ["ASTRA_DB_API_ENDPOINT"], "strict": True},
                    "token": {"type": "env_var", "env_vars": ["ASTRA_DB_APPLICATION_TOKEN"], "strict": True},
                    "collection_name": "documents",
                    "embedding_dimension": 768,
                    "duplicates_policy": "NONE",
                    "similarity": "cosine",
                },
            },
        },
    }
    retriever = AstraEmbeddingRetriever.from_dict(data)
    assert retriever.top_k == 42
    assert retriever.filters == {"bar": "baz"}
