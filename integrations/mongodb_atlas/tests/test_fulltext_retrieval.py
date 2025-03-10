# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from time import sleep
from typing import List, Union
from unittest.mock import MagicMock

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
        document_store.collection.delete_many({})
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

    def test_pipeline_correctly_passes_parameters(self):
        document_store = get_document_store()
        mock_collection = MagicMock()
        document_store._collection = mock_collection
        mock_collection.aggregate.return_value = []
        document_store._fulltext_retrieval(
            query=["spam", "eggs"],
            fuzzy={"maxEdits": 1},
            match_criteria="any",
            score={"boost": {"value": 3}},
            filters={"field": "meta.meta_field", "operator": "==", "value": "right_value"},
            top_k=5,
        )

        # Assert aggregate was called with the correct pipeline
        assert mock_collection.aggregate.called
        actual_pipeline = mock_collection.aggregate.call_args[0][0]
        expected_pipeline = [
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
                    "index": "full_text_index",
                }
            },
            {"$match": {"meta.meta_field": {"$eq": "right_value"}}},
            {"$limit": 5},
            {
                "$project": {
                    "_id": 0,
                    "blob": 1,
                    "content": 1,
                    "embedding": 1,
                    "meta": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        assert actual_pipeline == expected_pipeline

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

    @pytest.mark.parametrize("query", ["", []])
    def test_empty_query_raises_value_error(self, query: Union[str, List], document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            document_store._fulltext_retrieval(query=query)

    def test_empty_synonyms_raises_value_error(self, document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            document_store._fulltext_retrieval(query="fox", synonyms="")

    def test_synonyms_and_fuzzy_raises_value_error(self, document_store: MongoDBAtlasDocumentStore):
        with pytest.raises(ValueError):
            document_store._fulltext_retrieval(query="fox", synonyms="wolf", fuzzy={"maxEdits": 1})
