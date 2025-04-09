# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import FilterableDocsFixtureMixin
from haystack.utils import Secret
from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@patch("haystack_integrations.document_stores.mongodb_atlas.document_store.AsyncMongoClient")
def test_init_is_lazy(_mock_client):
    MongoDBAtlasDocumentStore(
        mongo_connection_string=Secret.from_token("test"),
        database_name="database_name",
        collection_name="collection_name",
        vector_search_index="cosine_index",
        full_text_search_index="full_text_index",
    )

    _mock_client.assert_not_called()


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING"),
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestDocumentStoreAsync(FilterableDocsFixtureMixin):
    @pytest.fixture
    def document_store(self):
        database_name = "haystack_integration_test"
        collection_name = "test_collection_" + str(uuid4())

        with MongoClient(
            os.environ["MONGO_CONNECTION_STRING"], driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
        ) as connection:
            database = connection[database_name]
            if collection_name in database.list_collection_names():
                database[collection_name].drop()
            database.create_collection(collection_name)
            database[collection_name].create_index("id", unique=True)

            store = MongoDBAtlasDocumentStore(
                database_name=database_name,
                collection_name=collection_name,
                vector_search_index="cosine_index",
                full_text_search_index="full_text_index",
            )
            yield store
            database[collection_name].drop()

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(content="some text")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    @pytest.mark.asyncio
    async def test_write_blob_async(self, document_store: MongoDBAtlasDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(blob=bytestream)]
        await document_store.write_documents_async(docs)
        retrieved_docs = await document_store.filter_documents_async()
        assert retrieved_docs == docs

    @pytest.mark.asyncio
    async def test_count_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(content="some text")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 1

    @pytest.mark.asyncio
    async def test_filter_documents_async(self, document_store: MongoDBAtlasDocumentStore, filterable_docs):
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
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(filters=filters)
        expected = [
            d
            for d in filterable_docs
            if (d.meta.get("number") == 100 and d.meta.get("chapter") == "intro")
            or (d.meta.get("page") == "90" and d.meta.get("chapter") == "conclusion")
        ]
        assert result == expected

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="some text")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 1
        await document_store.delete_documents_async(document_ids=["1"])
        assert await document_store.count_documents_async() == 0
