# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from uuid import uuid4

import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest, FilterDocumentsTest
from haystack.utils import Secret
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from pandas import DataFrame
from pymongo import MongoClient  # type: ignore
from pymongo.driver_info import DriverInfo  # type: ignore
import pandas as pd


@pytest.fixture
def document_store():
    database_name = "haystack_integration_test"
    collection_name = "test_collection_" + str(uuid4())

    connection: MongoClient = MongoClient(
        os.environ["MONGO_CONNECTION_STRING"], driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
    )
    database = connection[database_name]
    if collection_name in database.list_collection_names():
        database[collection_name].drop()
    database.create_collection(collection_name)
    database[collection_name].create_index("id", unique=True)

    store = MongoDBAtlasDocumentStore(
        database_name=database_name,
        collection_name=collection_name,
        vector_search_index="cosine_index",
    )
    yield store
    database[collection_name].drop()


@pytest.mark.skipif(
    "MONGO_CONNECTION_STRING" not in os.environ,
    reason="No MongoDB Atlas connection string provided",
)
class TestFilters(FilterDocumentsTest):
    pass