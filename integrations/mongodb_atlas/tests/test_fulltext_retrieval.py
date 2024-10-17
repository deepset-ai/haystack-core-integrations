# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@pytest.mark.skipif(
    "MONGO_CONNECTION_STRING" not in os.environ,
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestEmbeddingRetrieval:
    def test_basic_fulltext_retrieval(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_fulltext_collection",
            vector_search_index="default",
        )
        query = "crime"
        results = document_store._fulltext_retrieval(query=query)
        assert len(results) == 1

    def test_fulltext_retrieval_custom_path(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_fulltext_collection",
            vector_search_index="default",
        )
        query = "Godfather"
        path = "title"
        results = document_store._fulltext_retrieval(query=query, search_path=path)
        assert len(results) == 1

    def test_fulltext_retrieval_multi_paths_and_top_k(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_fulltext_collection",
            vector_search_index="default",
        )
        query = "movie"
        paths = ["title", "content"]
        results = document_store._fulltext_retrieval(query=query, search_path=paths)
        assert len(results) == 2

        results = document_store._fulltext_retrieval(query=query, search_path=paths, top_k=1)
        assert len(results) == 1


"""
[
    {
        "title": "The Matrix",
        "content": "A hacker discovers that his reality is a simulation in this movie.",
        "meta": {
            "author": "Wachowskis",
            "city": "San Francisco"
        }
    },
    {
        "title": "Inception",
        "content": "A thief who steals corporate secrets through the use of dream-sharing technology.",
        "meta": {
            "author": "Christopher Nolan",
            "city": "Los Angeles"
        }
    },
    {
        "title": "Interstellar",
        "content": "A team of explorers travel through a wormhole in space in an attempt
                        to ensure humanity's survival.",
        "meta": {
            "author": "Christopher Nolan",
            "city": "Houston"
        }
    },
    {
        "title": "The Dark Knight",
        "content": "When the menace known as the Joker emerges from his mysterious past,
                        he wreaks havoc on Gotham.",
        "meta": {
            "author": "Christopher Nolan",
            "city": "Gotham"
        }
    },
    {
        "title": "The Godfather Movie",
        "content": "The aging patriarch of an organized crime dynasty transfers
                        control of his empire to his reluctant son.",
        "meta": {
            "author": "Mario Puzo",
            "city": "New York"
        }
    }
]

"""
