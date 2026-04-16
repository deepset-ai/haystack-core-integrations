# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import FilterableDocsFixtureMixin

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
class TestDocumentStoreAsync(FilterableDocsFixtureMixin):
    @pytest.fixture
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
                await store.connection.close()

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

    async def test_count_documents_by_filter_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 4

        count_a = await document_store.count_documents_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count_a == 3

        count_active = await document_store.count_documents_by_filter_async(
            filters={"field": "meta.status", "operator": "==", "value": "active"}
        )
        assert count_active == 3

    async def test_count_unique_metadata_by_filter_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 5

        distinct_counts_a = await document_store.count_unique_metadata_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a["category"] == 1
        assert distinct_counts_a["status"] == 2
        # Category A docs have priorities 1, 1, 3 -> 2 unique values
        assert distinct_counts_a["priority"] == 2

    async def test_get_metadata_fields_info_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "number": 1, "ratio": 0.5}),
            Document(content="Doc 2", meta={"category": "B", "is_valid": True}),
        ]
        await document_store.write_documents_async(docs)

        fields_info = await document_store.get_metadata_fields_info_async()

        assert fields_info["content"]["type"] == "text"
        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["number"]["type"] == "long"
        assert fields_info["ratio"]["type"] == "float"
        assert fields_info["is_valid"]["type"] == "boolean"

    async def test_get_metadata_field_min_max_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"score": 10}),
            Document(content="Doc 2", meta={"score": 100}),
            Document(content="Doc 3", meta={"score": 20}),
        ]
        await document_store.write_documents_async(docs)

        result = await document_store.get_metadata_field_min_max_async("score")
        assert result["min"] == 10
        assert result["max"] == 100

    async def test_get_metadata_field_unique_values_async(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"tag": "alpha"}),
            Document(content="Doc 2", meta={"tag": "beta"}),
            Document(content="Doc 3", meta={"tag": "gamma"}),
            Document(content="Doc 4", meta={"tag": "alpha"}),
        ]
        await document_store.write_documents_async(docs)

        values, total_count = await document_store.get_metadata_field_unique_values_async("tag")
        assert total_count == 3
        assert sorted(values) == ["alpha", "beta", "gamma"]

        values_subset, count_subset = await document_store.get_metadata_field_unique_values_async(
            "tag", search_term="b", from_=0, size=10
        )
        assert count_subset == 1
        assert sorted(values_subset) == ["beta"]

        values_page, count_page = await document_store.get_metadata_field_unique_values_async("tag", from_=1, size=1)
        assert count_page == 3
        assert len(values_page) == 1
        assert values_page[0] in ["alpha", "beta", "gamma"]
