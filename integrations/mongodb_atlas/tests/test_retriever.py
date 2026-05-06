# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils.auth import EnvVarSecret

from haystack_integrations.components.retrievers.mongodb_atlas import (
    MongoDBAtlasEmbeddingRetriever,
    MongoDBAtlasFullTextRetriever,
)
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

EMBEDDING = {
    "retriever_cls": MongoDBAtlasEmbeddingRetriever,
    "retriever_type": "haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever",  # noqa: E501
    "collection_name": "test_embeddings_collection",
    "sync_method": "_embedding_retrieval",
    "async_method": "_embedding_retrieval_async",
    "doc": Document(content="Test doc", embedding=[0.1, 0.2]),
    "run_input": {"query_embedding": [0.3, 0.5]},
    "store_call_extras": {},
}

FULLTEXT = {
    "retriever_cls": MongoDBAtlasFullTextRetriever,
    "retriever_type": "haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever",  # noqa: E501
    "collection_name": "test_full_text_collection",
    "sync_method": "_fulltext_retrieval",
    "async_method": "_fulltext_retrieval_async",
    "doc": Document(content="Lorem ipsum"),
    "run_input": {"query": "Lorem ipsum"},
    "store_call_extras": {"fuzzy": None, "match_criteria": None, "score": None, "synonyms": None},
}


@pytest.mark.parametrize("case", [EMBEDDING, FULLTEXT], ids=["embedding", "fulltext"])
class TestRetriever:
    @pytest.fixture
    def mock_store(self):
        return Mock(spec=MongoDBAtlasDocumentStore)

    def test_init_raises_for_invalid_document_store(self, case):
        with pytest.raises(ValueError, match="MongoDBAtlasDocumentStore"):
            case["retriever_cls"](document_store=object())

    def test_init_default(self, case, mock_store):
        retriever = case["retriever_cls"](document_store=mock_store)
        assert retriever.document_store is mock_store
        assert retriever.filters == {}
        assert retriever.top_k == 10
        assert retriever.filter_policy == FilterPolicy.REPLACE

        retriever = case["retriever_cls"](document_store=mock_store, filter_policy="merge")
        assert retriever.filter_policy == FilterPolicy.MERGE

        with pytest.raises(ValueError):
            case["retriever_cls"](document_store=mock_store, filter_policy="wrong_policy")

    @pytest.mark.parametrize("filter_policy", [FilterPolicy.REPLACE, FilterPolicy.MERGE])
    def test_init_with_args(self, case, mock_store, filter_policy):
        filters = {"field": "meta.some_field", "operator": "==", "value": "SomeValue"}
        retriever = case["retriever_cls"](
            document_store=mock_store, filters=filters, top_k=5, filter_policy=filter_policy
        )
        assert retriever.document_store is mock_store
        assert retriever.filters == filters
        assert retriever.top_k == 5
        assert retriever.filter_policy == filter_policy

    def test_to_dict(self, case, monkeypatch):
        monkeypatch.setenv("MONGO_CONNECTION_STRING", "test_conn_str")
        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name=case["collection_name"],
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        )

        retriever = case["retriever_cls"](document_store=document_store, filters={"field": "value"}, top_k=5)

        assert retriever.to_dict() == {
            "type": case["retriever_type"],
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",  # noqa: E501
                    "init_parameters": {
                        "mongo_connection_string": {
                            "env_vars": ["MONGO_CONNECTION_STRING"],
                            "strict": True,
                            "type": "env_var",
                        },
                        "database_name": "haystack_integration_test",
                        "collection_name": case["collection_name"],
                        "vector_search_index": "cosine_index",
                        "full_text_search_index": "full_text_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
                "filter_policy": "replace",
            },
        }

    @pytest.mark.parametrize("include_filter_policy", [True, False], ids=["with_policy", "no_policy"])
    def test_from_dict(self, case, monkeypatch, include_filter_policy):
        monkeypatch.setenv("MONGO_CONNECTION_STRING", "test_conn_str")
        init_parameters = {
            "document_store": {
                "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
                "init_parameters": {
                    "mongo_connection_string": {
                        "env_vars": ["MONGO_CONNECTION_STRING"],
                        "strict": True,
                        "type": "env_var",
                    },
                    "database_name": "haystack_integration_test",
                    "collection_name": case["collection_name"],
                    "vector_search_index": "cosine_index",
                    "full_text_search_index": "full_text_index",
                },
            },
            "filters": {"field": "value"},
            "top_k": 5,
        }
        if include_filter_policy:
            init_parameters["filter_policy"] = "replace"

        retriever = case["retriever_cls"].from_dict(
            {"type": case["retriever_type"], "init_parameters": init_parameters}
        )

        document_store = retriever.document_store
        assert isinstance(document_store, MongoDBAtlasDocumentStore)
        assert isinstance(document_store.mongo_connection_string, EnvVarSecret)
        assert document_store.database_name == "haystack_integration_test"
        assert document_store.collection_name == case["collection_name"]
        assert document_store.vector_search_index == "cosine_index"
        assert document_store.full_text_search_index == "full_text_index"
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_run(self, case, mock_store):
        getattr(mock_store, case["sync_method"]).return_value = [case["doc"]]

        retriever = case["retriever_cls"](document_store=mock_store)
        res = retriever.run(**case["run_input"])

        getattr(mock_store, case["sync_method"]).assert_called_once_with(
            **case["run_input"], **case["store_call_extras"], filters={}, top_k=10
        )
        assert res == {"documents": [case["doc"]]}

    def test_run_merge_policy_filter(self, case, mock_store):
        getattr(mock_store, case["sync_method"]).return_value = [case["doc"]]
        init_filter = {"field": "meta.some_field", "operator": "==", "value": "SomeValue"}
        run_filter = {"field": "meta.some_field", "operator": "==", "value": "Test"}

        retriever = case["retriever_cls"](
            document_store=mock_store, filters=init_filter, filter_policy=FilterPolicy.MERGE
        )
        res = retriever.run(**case["run_input"], filters=run_filter)

        # as the both init and run filters are filtering the same field, the run filter takes precedence
        getattr(mock_store, case["sync_method"]).assert_called_once_with(
            **case["run_input"], **case["store_call_extras"], filters=run_filter, top_k=10
        )
        assert res == {"documents": [case["doc"]]}

    async def test_run_async(self, case, mock_store):
        getattr(mock_store, case["async_method"]).return_value = [case["doc"]]

        retriever = case["retriever_cls"](document_store=mock_store)
        res = await retriever.run_async(**case["run_input"])

        getattr(mock_store, case["async_method"]).assert_called_once_with(
            **case["run_input"], **case["store_call_extras"], filters={}, top_k=10
        )
        assert res == {"documents": [case["doc"]]}

    async def test_run_merge_policy_filter_async(self, case, mock_store):
        getattr(mock_store, case["async_method"]).return_value = [case["doc"]]
        init_filter = {"field": "meta.some_field", "operator": "==", "value": "SomeValue"}
        run_filter = {"field": "meta.some_field", "operator": "==", "value": "Test"}

        retriever = case["retriever_cls"](
            document_store=mock_store, filters=init_filter, filter_policy=FilterPolicy.MERGE
        )
        res = await retriever.run_async(**case["run_input"], filters=run_filter)

        # as the both init and run filters are filtering the same field, the run filter takes precedence
        getattr(mock_store, case["async_method"]).assert_called_once_with(
            **case["run_input"], **case["store_call_extras"], filters=run_filter, top_k=10
        )
        assert res == {"documents": [case["doc"]]}
