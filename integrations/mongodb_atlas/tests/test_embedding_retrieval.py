# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.document_stores.errors import DocumentStoreError

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestEmbeddingRetrievalUnit:
    def test_raises_for_empty_query_embedding(self, mocked_store_collection):
        store, _ = mocked_store_collection
        with pytest.raises(ValueError, match="Query embedding must not be empty"):
            store._embedding_retrieval(query_embedding=[])

    def test_wraps_exception_with_filters_hint(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.side_effect = RuntimeError("boom")
        with pytest.raises(DocumentStoreError, match="vector_search_index"):
            store._embedding_retrieval(
                query_embedding=[0.1, 0.2],
                filters={"field": "meta.f", "operator": "==", "value": "x"},
            )

    def test_uses_custom_embedding_field(self, mocked_store_collection):
        store, collection = mocked_store_collection
        store.embedding_field = "custom_vector"

        store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3])

        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0]["$vectorSearch"]["path"] == "custom_vector"


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING"),
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestEmbeddingRetrieval:
    @pytest.fixture
    def make_store(self):
        def _make(vector_search_index="cosine_index"):
            return MongoDBAtlasDocumentStore(
                database_name="haystack_integration_test",
                collection_name="test_embeddings_collection",
                vector_search_index=vector_search_index,
                full_text_search_index="full_text_index",
            )

        return _make

    def test_embedding_retrieval_cosine_similarity(self, make_store):
        results = make_store()._embedding_retrieval(query_embedding=[0.1] * 768, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Document C"
        assert results[1].content == "Document B"
        assert results[0].score > results[1].score

    def test_embedding_retrieval_dot_product(self, make_store):
        results = make_store("dotProduct_index")._embedding_retrieval(query_embedding=[0.1] * 768, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Document A"
        assert results[1].content == "Document B"
        assert results[0].score > results[1].score

    def test_embedding_retrieval_euclidean(self, make_store):
        results = make_store("euclidean_index")._embedding_retrieval(query_embedding=[0.1] * 768, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Document C"
        assert results[1].content == "Document B"
        assert results[0].score > results[1].score

    def test_query_embedding_wrong_dimension(self, make_store):
        with pytest.raises(DocumentStoreError):
            make_store()._embedding_retrieval(query_embedding=[0.1] * 4)

    def test_embedding_retrieval_with_filters(self, make_store):
        """
        Note: we can combine embedding retrieval with filters
        because the `cosine_index` vector_search_index was created with the `content` field as the filter field.
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
        filters = {"field": "content", "operator": "!=", "value": "Document A"}
        results = make_store()._embedding_retrieval(query_embedding=[0.1] * 768, top_k=2, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert doc.content != "Document A"
