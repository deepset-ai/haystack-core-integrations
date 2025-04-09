# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from time import sleep
from typing import List, Union

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


def get_document_store():
    return MongoDBAtlasDocumentStore(
        mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING_2"),
        database_name="haystack_test",
        collection_name="test_collection",
        vector_search_index="cosine_index",
        full_text_search_index="full_text_index",
    )


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING_2"),
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestFullTextRetrieval:

    @pytest.fixture(scope="class")
    def document_store(self) -> MongoDBAtlasDocumentStore:
        return get_document_store()

    @pytest.fixture(autouse=True, scope="class")
    def setup_teardown(self, document_store):
        document_store._ensure_connection_setup()
        document_store._collection.delete_many({})
        document_store.write_documents(
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

    @pytest.fixture(autouse=True)
    async def setup_async_connection(self, document_store):
        """
        Ensures that the async connection is set up in the same event loop as the test.
        This fixture is automatically used by all async tests.
        """
        await document_store._ensure_connection_setup_async()
        yield

    @pytest.mark.asyncio
    async def test_query_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        results = await document_store._fulltext_retrieval_async(query="fox", top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_fuzzy_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        results = await document_store._fulltext_retrieval_async(query="fax", fuzzy={"maxEdits": 1}, top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_filters_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        filters = {"field": "meta.meta_field", "operator": "==", "value": "right_value"}

        results = await document_store._fulltext_retrieval_async(query="fox", top_k=3, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
            assert doc.meta["meta_field"] == "right_value"

    @pytest.mark.asyncio
    async def test_synonyms_retrieval_async(self, document_store: MongoDBAtlasDocumentStore):
        results = await document_store._fulltext_retrieval_async(query="reynard", synonyms="synonym_mapping", top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["", []])
    async def test_empty_query_raises_value_error_async(
        self, query: Union[str, List], document_store: MongoDBAtlasDocumentStore
    ):
        with pytest.raises(ValueError):
            await document_store._fulltext_retrieval_async(query=query)

    @pytest.mark.asyncio
    async def test_empty_synonyms_raises_value_error_async(self, document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            await document_store._fulltext_retrieval_async(query="fox", synonyms="")

    @pytest.mark.asyncio
    async def test_synonyms_and_fuzzy_raises_value_error_async(self, document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            await document_store._fulltext_retrieval_async(query="fox", synonyms="wolf", fuzzy={"maxEdits": 1})
