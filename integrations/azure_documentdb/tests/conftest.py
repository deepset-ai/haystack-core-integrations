# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.utils import Secret

from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore


@pytest.fixture
def store_kwargs():
    return {
        "database_name": "test_db",
        "collection_name": "test_collection",
        "vector_search_index": "vector_index",
        "full_text_search_index": "text_index",
        "mongo_connection_string": Secret.from_token("mongodb://localhost:27017"),
    }


@pytest.fixture
def mocked_store_collection(store_kwargs):
    with patch("haystack_integrations.document_stores.azure_documentdb.document_store.MongoClient") as client_class:
        collection = MagicMock()
        collection.aggregate.return_value = []
        client = client_class.return_value
        database = MagicMock()
        client.__getitem__.return_value = database
        database.__getitem__.return_value = collection
        database.list_collection_names.return_value = ["test_collection"]
        client.admin.command.return_value = {"ok": 1}
        yield AzureDocumentDBDocumentStore(**store_kwargs), collection, client, database


@pytest.fixture
def mocked_store_collection_async(store_kwargs):
    with patch(
        "haystack_integrations.document_stores.azure_documentdb.document_store.AsyncMongoClient"
    ) as client_class:
        collection = MagicMock()
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[])
        collection.aggregate = AsyncMock(return_value=cursor)
        client = client_class.return_value
        database = MagicMock()
        client.__getitem__.return_value = database
        database.__getitem__.return_value = collection
        database.list_collection_names = AsyncMock(return_value=["test_collection"])
        client.admin.command = AsyncMock(return_value={"ok": 1})
        client.close = AsyncMock()
        yield AzureDocumentDBDocumentStore(**store_kwargs), collection, client, database
