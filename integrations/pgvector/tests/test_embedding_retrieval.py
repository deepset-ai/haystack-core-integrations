# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from haystack.dataclasses.document import Document
from numpy.random import rand

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
class TestEmbeddingRetrieval:
    @pytest.fixture
    def document_store_w_hnsw_index(self, request):
        connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
        table_name = f"haystack_hnsw_{request.node.name}"
        embedding_dimension = 768
        vector_function = "cosine_similarity"
        recreate_table = True
        search_strategy = "hnsw"

        store = PgvectorDocumentStore(
            connection_string=connection_string,
            table_name=table_name,
            embedding_dimension=embedding_dimension,
            vector_function=vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
        )
        yield store

        store.delete_table()

    @pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
    def test_embedding_retrieval_cosine_similarity(self, document_store: PgvectorDocumentStore):
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

        results = document_store._embedding_retrieval(
            query_embedding=query_embedding, top_k=2, filters={}, vector_function="cosine_similarity"
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document (cosine sim)"
        assert results[1].content == "2nd best document (cosine sim)"
        assert results[0].score > results[1].score

    @pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
    def test_embedding_retrieval_inner_product(self, document_store: PgvectorDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = rand(768).tolist()

        docs = [
            Document(content="Most similar document (inner product)", embedding=most_similar_embedding),
            Document(content="2nd best document (inner product)", embedding=second_best_embedding),
            Document(content="Not very similar document (inner product)", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store._embedding_retrieval(
            query_embedding=query_embedding, top_k=2, filters={}, vector_function="inner_product"
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document (inner product)"
        assert results[1].content == "2nd best document (inner product)"
        assert results[0].score > results[1].score

    @pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
    def test_embedding_retrieval_l2_distance(self, document_store: PgvectorDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.1] * 765 + [0.15] * 3
        second_best_embedding = [0.1] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = rand(768).tolist()

        docs = [
            Document(content="Most similar document (l2 dist)", embedding=most_similar_embedding),
            Document(content="2nd best document (l2 dist)", embedding=second_best_embedding),
            Document(content="Not very similar document (l2 dist)", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store._embedding_retrieval(
            query_embedding=query_embedding, top_k=2, filters={}, vector_function="l2_distance"
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document (l2 dist)"
        assert results[1].content == "2nd best document (l2 dist)"
        assert results[0].score < results[1].score

    @pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
    def test_embedding_retrieval_with_filters(self, document_store: PgvectorDocumentStore):
        docs = [Document(content=f"Document {i}", embedding=rand(768).tolist()) for i in range(10)]

        for i in range(10):
            docs[i].meta["meta_field"] = "custom_value" if i % 2 == 0 else "other_value"

        document_store.write_documents(docs)

        query_embedding = [0.1] * 768
        filters = {"field": "meta.meta_field", "operator": "==", "value": "custom_value"}

        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=3, filters=filters)
        assert len(results) == 3
        for result in results:
            assert result.meta["meta_field"] == "custom_value"
        assert results[0].score > results[1].score > results[2].score

    def test_empty_query_embedding(self, document_store: PgvectorDocumentStore):
        query_embedding: List[float] = []
        with pytest.raises(ValueError):
            document_store._embedding_retrieval(query_embedding=query_embedding)

    def test_query_embedding_wrong_dimension(self, document_store: PgvectorDocumentStore):
        query_embedding = [0.1] * 4
        with pytest.raises(ValueError):
            document_store._embedding_retrieval(query_embedding=query_embedding)
