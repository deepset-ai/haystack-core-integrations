# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

from haystack.dataclasses import Document
from haystack.utils.auth import EnvVarSecret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


class TestRetriever:
    def test_init_default(self, document_store: PgvectorDocumentStore):
        retriever = PgvectorEmbeddingRetriever(document_store=document_store)
        assert retriever.document_store == document_store
        assert retriever.filters == {}
        assert retriever.top_k == 10
        assert retriever.vector_function == document_store.vector_function

    def test_init(self, document_store: PgvectorDocumentStore):
        retriever = PgvectorEmbeddingRetriever(
            document_store=document_store, filters={"field": "value"}, top_k=5, vector_function="l2_distance"
        )
        assert retriever.document_store == document_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.vector_function == "l2_distance"

    def test_to_dict(self, document_store: PgvectorDocumentStore):
        retriever = PgvectorEmbeddingRetriever(
            document_store=document_store, filters={"field": "value"}, top_k=5, vector_function="l2_distance"
        )
        res = retriever.to_dict()
        t = "haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever"
        assert res == {
            "type": t,
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore",
                    "init_parameters": {
                        "connection_string": {"env_vars": ["PG_CONN_STR"], "strict": True, "type": "env_var"},
                        "table_name": "haystack_test_to_dict",
                        "embedding_dimension": 768,
                        "vector_function": "cosine_similarity",
                        "recreate_table": True,
                        "search_strategy": "exact_nearest_neighbor",
                        "hnsw_recreate_index_if_exists": False,
                        "hnsw_index_creation_kwargs": {},
                        "hnsw_ef_search": None,
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "vector_function": "l2_distance",
            },
        }

    def test_from_dict(self):
        t = "haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever"
        data = {
            "type": t,
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore",
                    "init_parameters": {
                        "connection_string": {"env_vars": ["PG_CONN_STR"], "strict": True, "type": "env_var"},
                        "table_name": "haystack_test_to_dict",
                        "embedding_dimension": 768,
                        "vector_function": "cosine_similarity",
                        "recreate_table": True,
                        "search_strategy": "exact_nearest_neighbor",
                        "hnsw_recreate_index_if_exists": False,
                        "hnsw_index_creation_kwargs": {},
                        "hnsw_ef_search": None,
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "vector_function": "l2_distance",
            },
        }

        retriever = PgvectorEmbeddingRetriever.from_dict(data)
        document_store = retriever.document_store

        assert isinstance(document_store, PgvectorDocumentStore)
        assert isinstance(document_store.connection_string, EnvVarSecret)
        assert document_store.table_name == "haystack_test_to_dict"
        assert document_store.embedding_dimension == 768
        assert document_store.vector_function == "cosine_similarity"
        assert document_store.recreate_table
        assert document_store.search_strategy == "exact_nearest_neighbor"
        assert not document_store.hnsw_recreate_index_if_exists
        assert document_store.hnsw_index_creation_kwargs == {}
        assert document_store.hnsw_ef_search is None

        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.vector_function == "l2_distance"

    def test_run(self):
        mock_store = Mock(spec=PgvectorDocumentStore)
        doc = Document(content="Test doc", embedding=[0.1, 0.2])
        mock_store._embedding_retrieval.return_value = [doc]

        retriever = PgvectorEmbeddingRetriever(document_store=mock_store, vector_function="l2_distance")
        res = retriever.run(query_embedding=[0.3, 0.5])

        mock_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.3, 0.5], filters={}, top_k=10, vector_function="l2_distance"
        )

        assert res == {"documents": [doc]}
