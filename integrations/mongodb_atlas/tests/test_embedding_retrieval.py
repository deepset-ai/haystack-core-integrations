# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

import pytest
from haystack.document_stores.errors import DocumentStoreError
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@pytest.mark.skipif(
    "MONGO_CONNECTION_STRING" not in os.environ,
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestEmbeddingRetrieval:
    def test_embedding_retrieval_cosine_similarity(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_embeddings_collection",
            vector_search_index="cosine_index",
        )
        query_embedding = [0.1] * 768
        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Document A"
        assert results[1].content == "Document B"
        assert results[0].score > results[1].score

    def test_embedding_retrieval_dot_product(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_embeddings_collection",
            vector_search_index="dotProduct_index",
        )
        query_embedding = [0.1] * 768
        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Document A"
        assert results[1].content == "Document B"
        assert results[0].score > results[1].score

    def test_embedding_retrieval_euclidean(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_embeddings_collection",
            vector_search_index="euclidean_index",
        )
        query_embedding = [0.1] * 768
        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Document C"
        assert results[1].content == "Document B"
        assert results[0].score > results[1].score

    def test_empty_query_embedding(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_embeddings_collection",
            vector_search_index="cosine_index",
        )
        query_embedding: List[float] = []
        with pytest.raises(ValueError):
            document_store._embedding_retrieval(query_embedding=query_embedding)

    def test_query_embedding_wrong_dimension(self):
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_embeddings_collection",
            vector_search_index="cosine_index",
        )
        query_embedding = [0.1] * 4
        with pytest.raises(DocumentStoreError):
            document_store._embedding_retrieval(query_embedding=query_embedding)

    def test_embedding_retrieval_with_filters(self):
        """
        Note: we can combine embedding retrieval with filters
        becuse the `cosine_index` vector_search_index was created with the `content` field as the filter field.
        {
        "fields": [
            {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,
            "similarity": "cosine"
            },
            {
            "type": "filter",
            "path": "content"
            }
        ]
        }
        """
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_embeddings_collection",
            vector_search_index="cosine_index",
        )
        query_embedding = [0.1] * 768
        filters = {"field": "content", "operator": "!=", "value": "Document A"}
        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert doc.content != "Document A"
