# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from time import sleep

import pytest
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestFullTextRetrievalUnit:
    @pytest.mark.parametrize("query", ["", []])
    def test_raises_for_empty_query(self, mocked_store_collection, query):
        store, _ = mocked_store_collection
        with pytest.raises(ValueError, match="query must not be empty"):
            store._fulltext_retrieval(query=query)

    def test_raises_for_empty_synonyms(self, mocked_store_collection):
        store, _ = mocked_store_collection
        with pytest.raises(ValueError, match="synonyms cannot be an empty string"):
            store._fulltext_retrieval(query="fox", synonyms="")

    def test_raises_when_synonyms_combined_with_fuzzy(self, mocked_store_collection):
        store, _ = mocked_store_collection
        with pytest.raises(ValueError, match="synonyms and fuzzy"):
            store._fulltext_retrieval(query="fox", synonyms="wolf", fuzzy={"maxEdits": 1})

    def test_wraps_exception_with_filters_hint(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.side_effect = RuntimeError("kapow")
        with pytest.raises(DocumentStoreError, match="full_text_search_index"):
            store._fulltext_retrieval(query="fox", filters={"field": "meta.f", "operator": "==", "value": "x"})

    def test_warns_when_synonyms_without_match_criteria(self, mocked_store_collection, caplog):
        store, _ = mocked_store_collection
        with caplog.at_level(logging.WARNING):
            store._fulltext_retrieval(query="fox", synonyms="syn")
        assert "matchCriteria" in caplog.text

    def test_pipeline_correctly_passes_parameters(self, mocked_store_collection):
        store, collection = mocked_store_collection

        store._fulltext_retrieval(
            query=["spam", "eggs"],
            fuzzy={"maxEdits": 1},
            match_criteria="any",
            score={"boost": {"value": 3}},
            filters={"field": "meta.meta_field", "operator": "==", "value": "right_value"},
            top_k=5,
        )

        # Assert aggregate was called with the correct pipeline
        assert collection.aggregate.called
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline == [
            {
                "$search": {
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "fuzzy": {"maxEdits": 1},
                                    "matchCriteria": "any",
                                    "path": "content",
                                    "query": ["spam", "eggs"],
                                    "score": {"boost": {"value": 3}},
                                }
                            }
                        ]
                    },
                    "index": "idx",
                }
            },
            {"$match": {"meta.meta_field": {"$eq": "right_value"}}},
            {"$limit": 5},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$project": {"_id": 0}},
        ]

        # Explicitly verify that the path in the text search is using the content_field
        assert pipeline[0]["$search"]["compound"]["must"][0]["text"]["path"] == store.content_field

    def test_pipeline_with_custom_content_field(self, mocked_store_collection):
        # Configure the store with a custom content field
        store, collection = mocked_store_collection
        store.content_field = "custom_text"

        # Execute the fulltext retrieval with the custom content field
        store._fulltext_retrieval(query="test query", top_k=3)

        # Assert aggregate was called with the correct pipeline
        assert collection.aggregate.called
        pipeline = collection.aggregate.call_args[0][0]

        # Verify the text search path is set to the custom content field
        # This is crucial - the path should use self.content_field, not be hardcoded to "content"
        assert pipeline[0]["$search"]["compound"]["must"][0]["text"]["path"] == "custom_text"

        # Verify the pipeline structure
        assert len(pipeline) == 5
        assert "$limit" in pipeline[2]
        assert "$addFields" in pipeline[3]
        assert "$project" in pipeline[4]


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING_2"),
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestFullTextRetrieval:
    @pytest.fixture(scope="class")
    def document_store(self) -> MongoDBAtlasDocumentStore:
        return MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING_2"),
            database_name="haystack_test",
            collection_name="test_collection",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        )

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

    def test_query_retrieval(self, document_store: MongoDBAtlasDocumentStore):
        results = document_store._fulltext_retrieval(query="fox", top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    def test_fuzzy_retrieval(self, document_store: MongoDBAtlasDocumentStore):
        results = document_store._fulltext_retrieval(query="fax", fuzzy={"maxEdits": 1}, top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score

    def test_filters_retrieval(self, document_store: MongoDBAtlasDocumentStore):
        filters = {"field": "meta.meta_field", "operator": "==", "value": "right_value"}

        results = document_store._fulltext_retrieval(query="fox", top_k=3, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
            assert doc.meta["meta_field"] == "right_value"

    def test_synonyms_retrieval(self, document_store: MongoDBAtlasDocumentStore):
        results = document_store._fulltext_retrieval(query="reynard", synonyms="synonym_mapping", top_k=2)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].score >= results[1].score
