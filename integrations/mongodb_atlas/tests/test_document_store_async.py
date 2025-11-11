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


@pytest.mark.skipif(not os.environ.get("MONGO_CONNECTION_STRING"), reason="No MongoDBAtlas connection string provided")
@pytest.mark.integration
class TestDocumentStoreAsync(FilterableDocsFixtureMixin):
    @pytest.fixture
    async def document_store(self):
        database_name = "haystack_integration_test"
        collection_name = "test_collection_" + str(uuid4())
        connection_string = os.environ["MONGO_CONNECTION_STRING"]

        # We're using the sync client for setup/teardown ease
        sync_client = MongoClient(connection_string, driver=DriverInfo(name="MongoDBAtlasHaystackIntegration"))

        store = MongoDBAtlasDocumentStore(
            database_name=database_name,
            collection_name=collection_name,
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        )
        try:
            database = sync_client[database_name]
            if collection_name in database.list_collection_names():
                database[collection_name].drop()
            database.create_collection(collection_name)
            database[collection_name].create_index("id", unique=True)

            # Initialize the async connection before yielding
            await store._ensure_connection_setup_async()

            yield store
        finally:
            # Ensure async connection is closed before synchronous teardown
            if store._connection_async:
                await store.connection.close()

            # Synchronous teardown
            if sync_client:
                try:
                    database = sync_client[database_name]
                    if collection_name in database.list_collection_names():
                        database[collection_name].drop()
                finally:
                    sync_client.close()

    async def test_write_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(content="some text")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    async def test_write_blob_async(self, document_store: MongoDBAtlasDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(blob=bytestream)]
        await document_store.write_documents_async(docs)
        retrieved_docs = await document_store.filter_documents_async()
        assert retrieved_docs == docs

    async def test_count_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(content="some text")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 1

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

    async def test_delete_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="some text")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 1
        await document_store.delete_documents_async(document_ids=["1"])
        assert await document_store.count_documents_async() == 0

    async def test_delete_by_filter_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Delete documents with category="A"
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        # Verify the remaining document is the one with category="B"
        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    async def test_update_by_filter_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update documents with category="A" to have status="published"
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 2

        # Verify the updated documents have the new metadata
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["status"] == "published"
            assert doc.meta["category"] == "A"

        # Verify documents with category="B" were not updated
        unpublished_docs = await document_store.filter_documents_async(
            filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert len(unpublished_docs) == 1
        assert "status" not in unpublished_docs[0].meta

    async def test_delete_all_documents_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="first doc"), Document(id="2", content="second doc")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2
        await document_store.delete_all_documents_async()
        assert await document_store.count_documents_async() == 0

    async def test_delete_all_documents_async_empty_collection(self, document_store: MongoDBAtlasDocumentStore):
        assert await document_store.count_documents_async() == 0
        await document_store.delete_all_documents_async()
        assert await document_store.count_documents_async() == 0

    async def test_delete_all_documents_async_with_recreate_collection(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="first doc"), Document(id="2", content="second doc")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        # Delete all documents with collection recreation
        await document_store.delete_all_documents_async(recreate_collection=True)
        assert await document_store.count_documents_async() == 0

        # Verify collection still exists and we can write to it
        new_docs = [Document(id="3", content="third doc")]
        await document_store.write_documents_async(new_docs)
        assert await document_store.count_documents_async() == 1
