# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack_integrations.components.retrievers.astra import AstraRetriever


@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN is not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_ID", "") == "", reason="ASTRA_DB_ID is not set")
@pytest.mark.integration
def test_retriever_to_json(document_store):
    retriever = AstraRetriever(document_store, filters={"foo": "bar"}, top_k=99)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.astra.retriever.AstraRetriever",
        "init_parameters": {
            "filters": {"foo": "bar"},
            "top_k": 99,
            "document_store": {
                "init_parameters": {
                    "astra_collection": "haystack_integration",
                    "astra_id": "63195634-ba44-49be-8a3c-12e830eb1c01",
                    "astra_keyspace": "astra_haystack_test",
                    "astra_region": "us-east-2",
                    "duplicates_policy": "OVERWRITE",
                    "embedding_dim": 768,
                    "similarity": "cosine",
                },
                "type": "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore",
            },
        },
    }


@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN is not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_ID", "") == "", reason="ASTRA_DB_ID is not set")
@pytest.mark.integration
def test_retriever_from_json():
    data = {
        "type": "haystack_integrations.components.retrievers.astra.retriever.AstraRetriever",
        "init_parameters": {
            "filters": {"bar": "baz"},
            "top_k": 42,
            "document_store": {
                "init_parameters": {
                    "astra_collection": "haystack_integration",
                    "astra_id": "63195634-ba44-49be-8a3c-12e830eb1c01",
                    "astra_application_token": os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""),
                    "astra_keyspace": "astra_haystack_test",
                    "astra_region": "us-east-2",
                    "duplicates_policy": "overwrite",
                    "embedding_dim": 768,
                    "similarity": "cosine",
                },
                "type": "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore",
            },
        },
    }
    retriever = AstraRetriever.from_dict(data)
    assert retriever.top_k == 42
    assert retriever.filters == {"bar": "baz"}
