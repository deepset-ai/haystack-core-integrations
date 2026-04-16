# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# The `cosine_index` used below is expected to be configured with a filter on the `content` field:
#   {
#     "fields": [
#       {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"},
#       {"type": "filter", "path": "content"}
#     ]
#   }
import os

import pytest
from haystack.document_stores.errors import DocumentStoreError

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


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

    def test_empty_query_embedding(self, make_store):
        with pytest.raises(ValueError):
            make_store()._embedding_retrieval(query_embedding=[])

    def test_query_embedding_wrong_dimension(self, make_store):
        with pytest.raises(DocumentStoreError):
            make_store()._embedding_retrieval(query_embedding=[0.1] * 4)

    def test_embedding_retrieval_with_filters(self, make_store):
        filters = {"field": "content", "operator": "!=", "value": "Document A"}
        results = make_store()._embedding_retrieval(query_embedding=[0.1] * 768, top_k=2, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert doc.content != "Document A"
