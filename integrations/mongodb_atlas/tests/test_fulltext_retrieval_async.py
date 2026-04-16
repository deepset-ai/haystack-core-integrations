# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from time import sleep
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING_2"), reason="No MongoDBAtlas connection string provided"
)
@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="class")
class TestFullTextRetrieval:
    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def document_store(self) -> MongoDBAtlasDocumentStore:
        store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING_2"),
            database_name="haystack_test",
            collection_name="test_collection",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        )
        await store._ensure_connection_setup_async()
        try:
            yield store
        finally:
            if store._connection_async:
                await store._connection_async.close()

    @pytest_asyncio.fixture(autouse=True, scope="class", loop_scope="class")
    async def setup_teardown(self, document_store):
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


class TestFullTextRetrievalUnit:
    async def test_pipeline_with_custom_content_field(self):
        store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
            content_field="custom_text",
        )
        store._collection_async = MagicMock()
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[])
        store._collection_async.aggregate = AsyncMock(return_value=cursor)
        store._ensure_connection_setup_async = AsyncMock()

        await store._fulltext_retrieval_async(query="test query", top_k=3)

        pipeline = store._collection_async.aggregate.call_args[0][0]
        # This is crucial - the path should use self.content_field, not be hardcoded to "content"
        assert pipeline[0]["$search"]["compound"]["must"][0]["text"]["path"] == "custom_text"

        # Verify the pipeline structure
        assert len(pipeline) == 5
        assert "$match" in pipeline[1]
        assert "$limit" in pipeline[2]
        assert "$addFields" in pipeline[3]
        assert "$project" in pipeline[4]
