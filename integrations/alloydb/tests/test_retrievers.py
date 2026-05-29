# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.utils.auth import EnvVarSecret

from haystack_integrations.components.retrievers.alloydb import AlloyDBEmbeddingRetriever, AlloyDBKeywordRetriever
from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore


class TestEmbeddingRetriever:
    def test_init_default(self, mock_store):
        retriever = AlloyDBEmbeddingRetriever(document_store=mock_store)
        assert retriever.document_store == mock_store
        assert retriever.filters == {}
        assert retriever.top_k == 10
        assert retriever.vector_function is None

    def test_init(self, mock_store):
        retriever = AlloyDBEmbeddingRetriever(
            document_store=mock_store,
            filters={"field": "value"},
            top_k=5,
            vector_function="l2_distance",
        )
        assert retriever.document_store == mock_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.vector_function == "l2_distance"

    def test_init_raises_with_wrong_store(self):
        with pytest.raises(ValueError, match="document_store must be an instance of AlloyDBDocumentStore"):
            AlloyDBEmbeddingRetriever(document_store="not_a_store")

    def test_to_dict(self, mock_store):
        retriever = AlloyDBEmbeddingRetriever(
            document_store=mock_store,
            filters={"field": "value"},
            top_k=5,
            vector_function="l2_distance",
        )
        res = retriever.to_dict()
        assert res["type"] == (
            "haystack_integrations.components.retrievers.alloydb.embedding_retriever.AlloyDBEmbeddingRetriever"
        )
        params = res["init_parameters"]
        assert params["filters"] == {"field": "value"}
        assert params["top_k"] == 5
        assert params["vector_function"] == "l2_distance"
        assert params["document_store"]["type"] == (
            "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore"
        )

    @pytest.mark.usefixtures("patches_for_unit_tests")
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv(
            "ALLOYDB_INSTANCE_URI",
            "projects/p/locations/r/clusters/c/instances/i",
        )
        monkeypatch.setenv("ALLOYDB_USER", "my-user")
        monkeypatch.setenv("ALLOYDB_PASSWORD", "my-password")

        data = {
            "type": "haystack_integrations.components.retrievers.alloydb.embedding_retriever.AlloyDBEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore",
                    "init_parameters": {
                        "instance_uri": {
                            "env_vars": ["ALLOYDB_INSTANCE_URI"],
                            "strict": True,
                            "type": "env_var",
                        },
                        "user": {"env_vars": ["ALLOYDB_USER"], "strict": True, "type": "env_var"},
                        "password": {
                            "env_vars": ["ALLOYDB_PASSWORD"],
                            "strict": False,
                            "type": "env_var",
                        },
                        "db": "postgres",
                        "enable_iam_auth": False,
                        "ip_type": "PRIVATE",
                        "create_extension": True,
                        "schema_name": "public",
                        "table_name": "haystack_test",
                        "language": "english",
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
            },
        }

        retriever = AlloyDBEmbeddingRetriever.from_dict(data)
        doc_store = retriever.document_store

        assert isinstance(doc_store, AlloyDBDocumentStore)
        assert isinstance(doc_store.instance_uri, EnvVarSecret)
        assert doc_store.table_name == "haystack_test"
        assert doc_store.embedding_dimension == 768
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.vector_function == "l2_distance"

    @pytest.mark.integration
    def test_run(self, document_store):
        docs = [Document(content="Python is great", embedding=[0.1] * 768)]
        document_store.write_documents(docs)

        retriever = AlloyDBEmbeddingRetriever(document_store=document_store, top_k=1)
        result = retriever.run(query_embedding=[0.1] * 768)

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Python is great"


class TestKeywordRetriever:
    def test_init_default(self, mock_store):
        retriever = AlloyDBKeywordRetriever(document_store=mock_store)
        assert retriever.document_store == mock_store
        assert retriever.filters == {}
        assert retriever.top_k == 10

    def test_init(self, mock_store):
        retriever = AlloyDBKeywordRetriever(
            document_store=mock_store,
            filters={"field": "value"},
            top_k=5,
        )
        assert retriever.document_store == mock_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5

    def test_init_raises_with_wrong_store(self):
        with pytest.raises(ValueError, match="document_store must be an instance of AlloyDBDocumentStore"):
            AlloyDBKeywordRetriever(document_store="not_a_store")

    def test_to_dict(self, mock_store):
        retriever = AlloyDBKeywordRetriever(
            document_store=mock_store,
            filters={"field": "value"},
            top_k=5,
        )
        res = retriever.to_dict()
        assert res["type"] == (
            "haystack_integrations.components.retrievers.alloydb.keyword_retriever.AlloyDBKeywordRetriever"
        )
        params = res["init_parameters"]
        assert params["filters"] == {"field": "value"}
        assert params["top_k"] == 5
        assert params["document_store"]["type"] == (
            "haystack_integrations.document_stores.alloydb.document_store.AlloyDBDocumentStore"
        )

    @pytest.mark.integration
    def test_run(self, document_store):
        docs = [Document(content="Python is a great programming language")]
        document_store.write_documents(docs)

        retriever = AlloyDBKeywordRetriever(document_store=document_store, top_k=1)
        result = retriever.run(query="Python programming")

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Python is a great programming language"
