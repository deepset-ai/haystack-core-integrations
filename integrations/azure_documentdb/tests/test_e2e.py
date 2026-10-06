# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from azure.core.credentials import AccessToken
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from pymongo import MongoClient

from haystack_integrations.components.retrievers.azure_documentdb import (
    AzureDocumentDBEmbeddingRetriever,
    AzureDocumentDBFullTextRetriever,
)
from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore
from haystack_integrations.document_stores.azure_documentdb.document_store import AzureIdentityTokenCallback

_DATABASE_NAME = "haystack_e2e"
_COLLECTION_NAME = "documents"
_VECTOR_INDEX_NAME = "haystack_vector_ivf"
_FULL_TEXT_INDEX_NAME = "haystack_content_bm25"


@dataclass
class StaticTokenCredential:
    """Token credential used to pass a short-lived Azure CLI token into the test process."""

    token: str

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:  # noqa: ARG002
        """Return the short-lived token supplied by the host Azure CLI session."""
        return AccessToken(self.token, int(time.time()) + 3600)


@pytest.fixture(scope="module")
def document_store() -> Iterator[AzureDocumentDBDocumentStore]:
    cluster_name = os.getenv("AZURE_DOCUMENTDB_CLUSTER_NAME")
    access_token = os.getenv("AZURE_DOCUMENTDB_ACCESS_TOKEN")
    if not cluster_name or not access_token:
        pytest.skip("Azure DocumentDB cluster name and access token are required for live E2E tests")

    credential = StaticTokenCredential(access_token)
    client = MongoClient(
        f"mongodb+srv://{cluster_name}.global.mongocluster.cosmos.azure.com/",
        authMechanism="MONGODB-OIDC",
        authMechanismProperties={"OIDC_CALLBACK": AzureIdentityTokenCallback(credential)},
        connectTimeoutMS=120000,
        retryWrites=False,
        tls=True,
    )
    database = client[_DATABASE_NAME]

    database.drop_collection(_COLLECTION_NAME)
    database.create_collection(_COLLECTION_NAME)
    database[_COLLECTION_NAME].create_index("id", unique=True)

    store = AzureDocumentDBDocumentStore(
        database_name=_DATABASE_NAME,
        collection_name=_COLLECTION_NAME,
        vector_search_index=_VECTOR_INDEX_NAME,
        full_text_search_index=_FULL_TEXT_INDEX_NAME,
        cluster_name=cluster_name,
        mongo_connection_string=None,
        azure_token_credential=credential,
    )
    try:
        yield store
    finally:
        store.close()
        if os.getenv("AZURE_DOCUMENTDB_KEEP_E2E_DATABASE", "false").lower() != "true":
            client.drop_database(_DATABASE_NAME)
        client.close()


def _retry_until_documents(operation: Callable[[], list[Document]], *, timeout: float = 180.0) -> list[Document]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            documents = operation()
            if documents:
                return documents
        except Exception as error:
            last_error = error
        time.sleep(2)
    if last_error:
        raise last_error
    msg = "Search index did not return documents before the timeout."
    raise AssertionError(msg)


@pytest.mark.integration
def test_crud_and_vector_e2e(document_store: AzureDocumentDBDocumentStore) -> None:
    documents = [
        Document(
            id="vector-guide",
            content="Azure DocumentDB provides integrated vector search for retrieval augmented generation.",
            embedding=[1.0, 0.0, 0.0],
            meta={"category": "guide", "year": 2026},
        ),
        Document(
            id="identity-guide",
            content="Microsoft Entra managed identity provides passwordless authentication to Azure services.",
            embedding=[0.0, 1.0, 0.0],
            meta={"category": "guide", "year": 2025},
        ),
        Document(
            id="unrelated-note",
            content="A short note about relational reporting workloads.",
            embedding=[0.0, 0.0, 1.0],
            meta={"category": "note", "year": 2024},
        ),
    ]

    assert document_store.write_documents(documents) == 3
    assert document_store.count_documents() == 3
    assert document_store.write_documents(documents, policy=DuplicatePolicy.SKIP) == 0

    filtered = document_store.filter_documents(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "guide"},
                {"field": "meta.year", "operator": ">=", "value": 2025},
            ],
        }
    )
    assert {document.id for document in filtered} == {"vector-guide", "identity-guide"}

    updated = Document(
        id="unrelated-note",
        content="A short note about relational reporting workloads.",
        embedding=[0.0, 0.0, 1.0],
        meta={"category": "reference", "year": 2024},
    )
    assert document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE) == 1
    assert document_store.filter_documents({"field": "meta.category", "operator": "==", "value": "reference"}) == [
        updated
    ]

    document_store.delete_documents(["unrelated-note"])
    assert document_store.count_documents() == 2

    document_store.create_vector_index(dimensions=3, kind="vector-ivf", similarity="COS", numLists=1)
    retriever = AzureDocumentDBEmbeddingRetriever(document_store=document_store, top_k=1)
    results = _retry_until_documents(lambda: retriever.run(query_embedding=[0.99, 0.01, 0.0])["documents"])
    assert results[0].id == "vector-guide"
    assert results[0].score is not None


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("AZURE_DOCUMENTDB_FULL_TEXT_SEARCH_ENABLED", "false").lower() != "true",
    reason="Azure DocumentDB BM25 full-text search is a gated preview.",
)
def test_full_text_e2e(document_store: AzureDocumentDBDocumentStore) -> None:
    document_store.write_documents(
        [Document(id="vector-guide", content="Azure DocumentDB provides integrated vector search.")],
        policy=DuplicatePolicy.OVERWRITE,
    )
    connection = document_store.connection
    assert isinstance(connection, MongoClient)
    assert document_store.full_text_search_index is not None
    connection[document_store.database_name].command(
        {
            "createSearchIndexes": document_store.collection_name,
            "indexes": [
                {
                    "name": document_store.full_text_search_index,
                    "definition": {
                        "mappings": {
                            "dynamic": False,
                            "fields": {document_store.content_field: {"type": "string"}},
                        }
                    },
                }
            ],
        }
    )
    retriever = AzureDocumentDBFullTextRetriever(document_store=document_store, top_k=2)
    results = _retry_until_documents(lambda: retriever.run(query="integrated vector search")["documents"])
    assert results[0].id == "vector-guide"
    assert results[0].score is not None
