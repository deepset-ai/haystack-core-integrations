# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from uuid import uuid4
import os

import pytest
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from numpy.random import rand


@pytest.fixture
def document_store(request):
    store = MongoDBAtlasDocumentStore(
        database_name="haystack_integration_test",
        collection_name="test_collection",
        vector_search_index="vector_index",
        recreate_collection=True,
    )
    return store


@pytest.mark.skipif(
    "MONGO_CONNECTION_STRING" not in os.environ,
    reason="No MongoDB Atlas connection string provided",
)
class TestEmbeddingRetrieval:
    
    def test_embedding_retrieval_cosine_similarity(self, document_store: MongoDBAtlasDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = rand(768).tolist()

        docs = [
            Document(content="Most similar document (cosine sim)", embedding=most_similar_embedding),
            Document(content="2nd best document (cosine sim)", embedding=second_best_embedding),
            Document(content="Not very similar document (cosine sim)", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store.embedding_retrieval(
            query_embedding=query_embedding, top_k=2, filters={}, similarity="cosine"
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document (cosine sim)"
        assert results[1].content == "2nd best document (cosine sim)"
        assert results[0].score > results[1].score

    def test_embedding_retrieval_dot_product(self, document_store: MongoDBAtlasDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = rand(768).tolist()

        docs = [
            Document(content="Most similar document (dot product)", embedding=most_similar_embedding),
            Document(content="2nd best document (dot product)", embedding=second_best_embedding),
            Document(content="Not very similar document (dot product)", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store.embedding_retrieval(
            query_embedding=query_embedding, top_k=2, filters={}, similarity="dotProduct"
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document (dot product)"
        assert results[1].content == "2nd best document (dot product)"
        assert results[0].score > results[1].score

    
    def test_embedding_retrieval_euclidean(self, document_store: MongoDBAtlasDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = rand(768).tolist()

        docs = [
            Document(content="Most similar document (euclidean)", embedding=most_similar_embedding),
            Document(content="2nd best document (euclidean)", embedding=second_best_embedding),
            Document(content="Not very similar document (euclidean)", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store.embedding_retrieval(
            query_embedding=query_embedding, top_k=2, filters={}, similarity="euclidean"
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document (euclidean)"
        assert results[1].content == "2nd best document (euclidean)"
        assert results[0].score > results[1].score

    def test_empty_query_embedding(self, document_store: MongoDBAtlasDocumentStore):
        query_embedding: List[float] = []
        with pytest.raises(ValueError):
            document_store.embedding_retrieval(query_embedding=query_embedding)

    def test_query_embedding_wrong_dimension(self, document_store: MongoDBAtlasDocumentStore):
        query_embedding = [0.1] * 4
        with pytest.raises(ValueError):
            document_store.embedding_retrieval(query_embedding=query_embedding)
