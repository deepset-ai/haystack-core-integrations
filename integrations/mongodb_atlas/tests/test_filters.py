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
    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.page", "operator": "==", "value": "90"},
                        {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
                    ],
                },
            ],
        }

        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.meta.get("number") == 100 and d.meta.get("chapter") == "intro")
                or (d.meta.get("page") == "90" and d.meta.get("chapter") == "conclusion")
            ],
        )