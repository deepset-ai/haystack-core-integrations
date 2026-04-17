# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from haystack.utils.auth import Secret
from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

_STORE_KWARGS = {
    "mongo_connection_string": Secret.from_token("mongodb://localhost:27017"),
    "database_name": "test_db",
    "collection_name": "test_collection",
    "vector_search_index": "idx",
    "full_text_search_index": "idx",
}


@pytest.fixture
def local_store():
    return MongoDBAtlasDocumentStore(**_STORE_KWARGS)


@pytest.fixture
def mocked_store_collection():
    with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_cls:
        collection = MagicMock()
        collection.aggregate.return_value = []
        client = mock_cls.return_value
        db = MagicMock()
        client.__getitem__.return_value = db
        db.__getitem__.return_value = collection
        db.list_collection_names.return_value = ["test_collection"]
        client.admin.command.return_value = {"ok": 1}
        yield MongoDBAtlasDocumentStore(**_STORE_KWARGS), collection


@pytest.fixture
def mocked_store_collection_async():
    with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.AsyncMongoClient") as mock_cls:
        collection = MagicMock()
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[])
        collection.aggregate = AsyncMock(return_value=cursor)
        client = mock_cls.return_value
        db = MagicMock()
        client.__getitem__.return_value = db
        db.__getitem__.return_value = collection
        db.list_collection_names = AsyncMock(return_value=["test_collection"])
        client.admin.command = AsyncMock(return_value={"ok": 1})
        yield MongoDBAtlasDocumentStore(**_STORE_KWARGS), collection


@pytest.fixture
def real_collection():
    database_name = "haystack_integration_test"
    collection_name = "test_collection_" + str(uuid4())
    client = MongoClient(
        os.environ["MONGO_CONNECTION_STRING"], driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
    )
    database = client[database_name]
    if collection_name in database.list_collection_names():
        database[collection_name].drop()
    database.create_collection(collection_name)
    database[collection_name].create_index("id", unique=True)
    try:
        yield database_name, collection_name, client
    finally:
        database[collection_name].drop()
        client.close()
