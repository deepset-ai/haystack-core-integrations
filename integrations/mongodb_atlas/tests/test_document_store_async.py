# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestEnsureConnectionSetupAsync:
    async def test_raises_when_ping_fails(self, local_store):
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.AsyncMongoClient") as mock_cls:
            mock_cls.return_value.admin.command = AsyncMock(side_effect=RuntimeError("nope"))
            with pytest.raises(DocumentStoreError, match="Connection to MongoDB Atlas failed"):
                await local_store._ensure_connection_setup_async()

    async def test_raises_when_collection_missing(self, local_store):
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.AsyncMongoClient") as mock_cls:
            client = mock_cls.return_value
            client.admin.command = AsyncMock(return_value={"ok": 1})
            db = MagicMock()
            db.list_collection_names = AsyncMock(return_value=["other_collection"])
            client.__getitem__.return_value = db
            with pytest.raises(DocumentStoreError, match="does not exist"):
                await local_store._ensure_connection_setup_async()


class TestMongoDBDocumentStoreAsyncUnit:
    async def test_get_metadata_fields_info(self, mocked_store_collection_async):
        store, collection = mocked_store_collection_async
        cursor = MagicMock()
        collection.find.return_value = cursor
        cursor.sort.return_value = cursor
        cursor.limit.return_value = cursor
        cursor.to_list = AsyncMock(
            return_value=[
                {"meta": {"category": "A", "number": 1, "ratio": 0.5}},
                {"meta": {"category": "B", "is_valid": True}},
            ]
        )

        fields_info = await store.get_metadata_fields_info_async()

        assert fields_info["content"] == {"type": "text"}
        assert fields_info["category"] == {"type": "keyword"}
        assert fields_info["number"] == {"type": "long"}
        assert fields_info["ratio"] == {"type": "float"}
        assert fields_info["is_valid"] == {"type": "boolean"}

    async def test_get_metadata_field_min_max(self, mocked_store_collection_async):
        store, collection = mocked_store_collection_async
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[{"min": 10, "max": 100}])
        collection.aggregate = AsyncMock(return_value=cursor)

        result = await store.get_metadata_field_min_max_async("number")

        assert result == {"min": 10, "max": 100}
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"]["min"] == {"$min": "$meta.number"}

    async def test_get_metadata_field_unique_values(self, mocked_store_collection_async):
        store, collection = mocked_store_collection_async
        cursor = MagicMock()
        cursor.to_list = AsyncMock(
            return_value=[{"count": [{"count": 5}], "values": [{"_id": "val1"}, {"_id": "val2"}]}]
        )
        collection.aggregate = AsyncMock(return_value=cursor)

        values, count = await store.get_metadata_field_unique_values_async("category", search_term="val", size=2)

        assert values == ["val1", "val2"]
        assert count == 5
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"] == {"_id": "$meta.category"}
        assert pipeline[1]["$match"] == {"_id": {"$regex": "val", "$options": "i"}}
        assert pipeline[2]["$facet"]["values"][2]["$limit"] == 2


@pytest.mark.skipif(not os.environ.get("MONGO_CONNECTION_STRING"), reason="No MongoDBAtlas connection string provided")
@pytest.mark.integration
class TestDocumentStoreAsync(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @pytest_asyncio.fixture
    async def document_store(self, real_collection):
        database_name, collection_name, _ = real_collection
        store = MongoDBAtlasDocumentStore(
            database_name=database_name,
            collection_name=collection_name,
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        )
        await store._ensure_connection_setup_async()
        try:
            yield store
        finally:
            if store._connection_async:
                await store._connection_async.close()

    async def test_count_not_empty_async(self, document_store):
        # Override needed: base class uses @staticmethod which breaks fixture injection
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

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

    async def test_delete_all_documents_async_with_recreate_collection(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="first doc"), Document(id="2", content="second doc")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async(recreate_collection=True)
        assert await document_store.count_documents_async() == 0

        new_docs = [Document(id="3", content="third doc")]
        await document_store.write_documents_async(new_docs)
        assert await document_store.count_documents_async() == 1
