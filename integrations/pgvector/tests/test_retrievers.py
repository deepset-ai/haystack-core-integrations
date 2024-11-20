# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils.auth import EnvVarSecret

from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


class TestEmbeddingRetriever:
    def test_init_default(self, mock_store):
        retriever = PgvectorEmbeddingRetriever(document_store=mock_store)
        assert retriever.document_store == mock_store
        assert retriever.filters == {}
        assert retriever.top_k == 10
        assert retriever.filter_policy == FilterPolicy.REPLACE
        assert retriever.vector_function == mock_store.vector_function

        retriever = PgvectorEmbeddingRetriever(document_store=mock_store, filter_policy="merge")
        assert retriever.filter_policy == FilterPolicy.MERGE

        with pytest.raises(ValueError):
            PgvectorEmbeddingRetriever(document_store=mock_store, filter_policy="invalid")

    def test_init(self, mock_store):
        retriever = PgvectorEmbeddingRetriever(
            document_store=mock_store, filters={"field": "value"}, top_k=5, vector_function="l2_distance"
        )
        assert retriever.document_store == mock_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.filter_policy == FilterPolicy.REPLACE
        assert retriever.vector_function == "l2_distance"

    def test_to_dict(self, mock_store):
        retriever = PgvectorEmbeddingRetriever(
            document_store=mock_store, filters={"field": "value"}, top_k=5, vector_function="l2_distance"
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
                        "schema_name": "public",
                        "table_name": "haystack",
                        "embedding_dimension": 768,
                        "vector_function": "cosine_similarity",
                        "recreate_table": True,
                        "search_strategy": "exact_nearest_neighbor",
                        "hnsw_recreate_index_if_exists": False,
                        "language": "english",
                        "hnsw_index_creation_kwargs": {},
                        "hnsw_index_name": "haystack_hnsw_index",
                        "hnsw_ef_search": None,
                        "keyword_index_name": "haystack_keyword_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "vector_function": "l2_distance",
                "filter_policy": "replace",
            },
        }

    @pytest.mark.usefixtures("patches_for_unit_tests")
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("PG_CONN_STR", "some-connection-string")
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
                        "hnsw_index_name": "haystack_hnsw_index",
                        "hnsw_ef_search": None,
                        "keyword_index_name": "haystack_keyword_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "vector_function": "l2_distance",
                "filter_policy": "replace",
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
        assert document_store.hnsw_index_name == "haystack_hnsw_index"
        assert document_store.hnsw_ef_search is None
        assert document_store.keyword_index_name == "haystack_keyword_index"

        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.filter_policy == FilterPolicy.REPLACE
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


class TestKeywordRetriever:
    def test_init_default(self, mock_store):
        retriever = PgvectorKeywordRetriever(document_store=mock_store)
        assert retriever.document_store == mock_store
        assert retriever.filters == {}
        assert retriever.top_k == 10

        retriever = PgvectorKeywordRetriever(document_store=mock_store, filter_policy="merge")
        assert retriever.filter_policy == FilterPolicy.MERGE

        with pytest.raises(ValueError):
            PgvectorKeywordRetriever(document_store=mock_store, filter_policy="invalid")

    def test_init(self, mock_store):
        retriever = PgvectorKeywordRetriever(document_store=mock_store, filters={"field": "value"}, top_k=5)
        assert retriever.document_store == mock_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5

    def test_init_with_filter_policy(self, mock_store):
        retriever = PgvectorKeywordRetriever(
            document_store=mock_store, filters={"field": "value"}, top_k=5, filter_policy=FilterPolicy.MERGE
        )
        assert retriever.document_store == mock_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.filter_policy == FilterPolicy.MERGE

    def test_to_dict(self, mock_store):
        retriever = PgvectorKeywordRetriever(document_store=mock_store, filters={"field": "value"}, top_k=5)
        res = retriever.to_dict()
        t = "haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever"
        assert res == {
            "type": t,
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore",
                    "init_parameters": {
                        "connection_string": {"env_vars": ["PG_CONN_STR"], "strict": True, "type": "env_var"},
                        "schema_name": "public",
                        "table_name": "haystack",
                        "embedding_dimension": 768,
                        "vector_function": "cosine_similarity",
                        "recreate_table": True,
                        "search_strategy": "exact_nearest_neighbor",
                        "hnsw_recreate_index_if_exists": False,
                        "language": "english",
                        "hnsw_index_creation_kwargs": {},
                        "hnsw_index_name": "haystack_hnsw_index",
                        "hnsw_ef_search": None,
                        "keyword_index_name": "haystack_keyword_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "filter_policy": "replace",
            },
        }

    @pytest.mark.usefixtures("patches_for_unit_tests")
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("PG_CONN_STR", "some-connection-string")
        t = "haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever"
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
                        "hnsw_index_name": "haystack_hnsw_index",
                        "hnsw_ef_search": None,
                        "keyword_index_name": "haystack_keyword_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "filter_policy": "replace",
            },
        }

        retriever = PgvectorKeywordRetriever.from_dict(data)
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
        assert document_store.hnsw_index_name == "haystack_hnsw_index"
        assert document_store.hnsw_ef_search is None
        assert document_store.keyword_index_name == "haystack_keyword_index"

        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5

    @pytest.mark.usefixtures("patches_for_unit_tests")
    def test_from_dict_without_filter_policy(self, monkeypatch):
        monkeypatch.setenv("PG_CONN_STR", "some-connection-string")
        t = "haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever"
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
                        "hnsw_index_name": "haystack_hnsw_index",
                        "hnsw_ef_search": None,
                        "keyword_index_name": "haystack_keyword_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
            },
        }

        retriever = PgvectorKeywordRetriever.from_dict(data)
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
        assert document_store.hnsw_index_name == "haystack_hnsw_index"
        assert document_store.hnsw_ef_search is None
        assert document_store.keyword_index_name == "haystack_keyword_index"

        assert retriever.filters == {"field": "value"}
        assert retriever.filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE
        assert retriever.top_k == 5

    def test_run(self):
        mock_store = Mock(spec=PgvectorDocumentStore)
        doc = Document(content="Test doc", embedding=[0.1, 0.2])
        mock_store._keyword_retrieval.return_value = [doc]

        retriever = PgvectorKeywordRetriever(document_store=mock_store)
        res = retriever.run(query="test query")

        mock_store._keyword_retrieval.assert_called_once_with(query="test query", filters={}, top_k=10)

        assert res == {"documents": [doc]}

    def test_run_with_filters(self):
        mock_store = Mock(spec=PgvectorDocumentStore)
        doc = Document(content="Test doc", embedding=[0.1, 0.2])
        mock_store._keyword_retrieval.return_value = [doc]

        retriever = PgvectorKeywordRetriever(
            document_store=mock_store, filter_policy=FilterPolicy.MERGE, filters={"field": "value"}
        )
        res = retriever.run(query="test query", filters={"field2": "value2"})

        mock_store.mock_calls[0].assert_called_once_with(
            query="test query", filters={"field": "value", "field2": "value2"}, top_k=10
        )

        assert res == {"documents": [doc]}
