# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from time import sleep
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class AsyncDocumentStoreContext:
    """Context manager for MongoDB Atlas document store with async support."""

    def __init__(
        self,
        mongo_connection_string,
        database_name,
        collection_name,
        vector_search_index,
        full_text_search_index,
        **kwargs,
    ):
        self.mongo_connection_string = mongo_connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.vector_search_index = vector_search_index
        self.full_text_search_index = full_text_search_index
        self.kwargs = kwargs
        self.store = None

    async def __aenter__(self):
        self.store = MongoDBAtlasDocumentStore(
            mongo_connection_string=self.mongo_connection_string,
            database_name=self.database_name,
            collection_name=self.collection_name,
            vector_search_index=self.vector_search_index,
            full_text_search_index=self.full_text_search_index,
            **self.kwargs,
        )
        await self.store._ensure_connection_setup_async()
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.store and self.store._connection_async:
            await self.store._connection_async.close()


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING_2"), reason="No MongoDBAtlas connection string provided"
)
@pytest.mark.integration
class TestFullTextRetrieval:
    @pytest.fixture
    async def document_store(self) -> MongoDBAtlasDocumentStore:
        async with AsyncDocumentStoreContext(
            mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING_2"),
            database_name="haystack_test",
            collection_name="test_collection",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        ) as store:
            yield store

    @pytest.fixture
    async def setup(self, document_store):
        # clean up the collection and insert test documents
        await document_store._collection_async.delete_many({})
        await document_store.write_documents_async(
            [
                Document(content="The quick brown fox chased the dog", meta={"meta_field": "right_value"}),
                Document(content="The fox was brown", meta={"meta_field": "right_value"}),
                Document(content="The lazy dog"),
                Document(content="fox fox fox"),
            ]
        )

        # Wait for documents to be indexed
        sleep(5)

        yield

    async def test_query_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        results = await document_store._fulltext_retrieval_async(query="fox", top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    async def test_fuzzy_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        results = await document_store._fulltext_retrieval_async(query="fax", fuzzy={"maxEdits": 1}, top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    async def test_filters_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        filters = {"field": "meta.meta_field", "operator": "==", "value": "right_value"}
        results = await document_store._fulltext_retrieval_async(query="fox", top_k=3, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
            assert doc.meta["meta_field"] == "right_value"

    async def test_synonyms_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        results = await document_store._fulltext_retrieval_async(query="reynard", synonyms="synonym_mapping", top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    @pytest.mark.parametrize("query", ["", []])
    async def test_empty_query_raises_value_error_async(
        self, query: str | list, document_store: MongoDBAtlasDocumentStore
    ):
        with pytest.raises(ValueError):
            await document_store._fulltext_retrieval_async(query=query)

    async def test_empty_synonyms_raises_value_error_async(self, document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            await document_store._fulltext_retrieval_async(query="fox", synonyms="")

    async def test_synonyms_and_fuzzy_raises_value_error_async(self, document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            await document_store._fulltext_retrieval_async(query="fox", synonyms="wolf", fuzzy={"maxEdits": 1})


class TestFullTextRetrievalUnitTests:
    @pytest.mark.asyncio
    @patch("haystack_integrations.document_stores.mongodb_atlas.document_store.AsyncMongoClient")
    async def test_pipeline_with_custom_content_field_async(self, mock_client, monkeypatch):
        """Test that custom content_field is correctly used in text search path for async fulltext retrieval."""
        # Set up the mock client
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        # Mock the collection names for the _collection_exists_async check
        mock_db.list_collection_names = AsyncMock(return_value=["test_collection"])

        # Setup the collection's aggregate method
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)

        # Create document store with custom content field
        document_store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("dummy_token"),
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
            content_field="custom_text",
        )

        # Use monkeypatch to replace the _ensure_connection_setup_async method
        monkeypatch.setattr(document_store, "_ensure_connection_setup_async", AsyncMock())

        # Set the mocked collection
        document_store._connection_async = mock_client.return_value
        document_store._collection_async = mock_collection

        # Execute the fulltext retrieval with the custom content field
        await document_store._fulltext_retrieval_async(query="test query", top_k=3)

        # Assert aggregate was called
        assert mock_collection.aggregate.called

        # Get the pipeline that was passed to aggregate
        call_args = mock_collection.aggregate.call_args
        actual_pipeline = call_args[0][0]

        # Verify the text search path is set to the custom content field
        assert actual_pipeline[0]["$search"]["compound"]["must"][0]["text"]["path"] == "custom_text"
        assert actual_pipeline[0]["$search"]["compound"]["must"][0]["text"]["path"] == document_store.content_field

        # Verify the pipeline structure
        assert len(actual_pipeline) == 5
        assert "$match" in actual_pipeline[1]
        assert "$limit" in actual_pipeline[2]
        assert "$addFields" in actual_pipeline[3]
        assert "$project" in actual_pipeline[4]
