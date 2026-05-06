# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils.auth import EnvVarSecret

from haystack_integrations.components.retrievers.supabase import SupabasePgvectorKeywordRetriever
from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore


def test_init_default(mock_store):
    retriever = SupabasePgvectorKeywordRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE


def test_init_filter_policy_string(mock_store):
    retriever = SupabasePgvectorKeywordRetriever(document_store=mock_store, filter_policy="merge")
    assert retriever.filter_policy == FilterPolicy.MERGE


def test_init_invalid_filter_policy(mock_store):
    with pytest.raises(ValueError):
        SupabasePgvectorKeywordRetriever(document_store=mock_store, filter_policy="invalid")


def test_init(mock_store):
    retriever = SupabasePgvectorKeywordRetriever(document_store=mock_store, filters={"field": "value"}, top_k=5)
    assert retriever.document_store == mock_store
    assert retriever.filters == {"field": "value"}
    assert retriever.top_k == 5


def test_init_with_filter_policy(mock_store):
    retriever = SupabasePgvectorKeywordRetriever(
        document_store=mock_store, filters={"field": "value"}, top_k=5, filter_policy=FilterPolicy.MERGE
    )
    assert retriever.document_store == mock_store
    assert retriever.filters == {"field": "value"}
    assert retriever.top_k == 5
    assert retriever.filter_policy == FilterPolicy.MERGE


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="must be an instance of SupabasePgvectorDocumentStore"):
        SupabasePgvectorKeywordRetriever(document_store="not a store")


def test_to_dict(mock_store):
    retriever = SupabasePgvectorKeywordRetriever(document_store=mock_store, filters={"field": "value"}, top_k=5)
    res = retriever.to_dict()
    t = "haystack_integrations.components.retrievers.supabase.keyword_retriever.SupabasePgvectorKeywordRetriever"
    assert res == {
        "type": t,
        "init_parameters": {
            "document_store": {
                "type": ("haystack_integrations.document_stores.supabase.document_store.SupabasePgvectorDocumentStore"),
                "init_parameters": {
                    "connection_string": {"env_vars": ["SUPABASE_DB_URL"], "strict": True, "type": "env_var"},
                    "create_extension": False,
                    "schema_name": "public",
                    "table_name": "haystack",
                    "embedding_dimension": 768,
                    "vector_type": "vector",
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
def test_from_dict(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some-connection-string")
    t = "haystack_integrations.components.retrievers.supabase.keyword_retriever.SupabasePgvectorKeywordRetriever"
    data = {
        "type": t,
        "init_parameters": {
            "document_store": {
                "type": ("haystack_integrations.document_stores.supabase.document_store.SupabasePgvectorDocumentStore"),
                "init_parameters": {
                    "connection_string": {"env_vars": ["SUPABASE_DB_URL"], "strict": True, "type": "env_var"},
                    "create_extension": False,
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

    retriever = SupabasePgvectorKeywordRetriever.from_dict(data)
    document_store = retriever.document_store

    assert isinstance(document_store, SupabasePgvectorDocumentStore)
    assert isinstance(document_store.connection_string, EnvVarSecret)
    assert not document_store.create_extension
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


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_from_dict_without_filter_policy(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some-connection-string")
    t = "haystack_integrations.components.retrievers.supabase.keyword_retriever.SupabasePgvectorKeywordRetriever"
    data = {
        "type": t,
        "init_parameters": {
            "document_store": {
                "type": ("haystack_integrations.document_stores.supabase.document_store.SupabasePgvectorDocumentStore"),
                "init_parameters": {
                    "connection_string": {"env_vars": ["SUPABASE_DB_URL"], "strict": True, "type": "env_var"},
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

    retriever = SupabasePgvectorKeywordRetriever.from_dict(data)
    document_store = retriever.document_store

    assert isinstance(document_store, SupabasePgvectorDocumentStore)
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


def test_run():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._keyword_retrieval.return_value = [doc]

    retriever = SupabasePgvectorKeywordRetriever(document_store=mock_store)
    res = retriever.run(query="test query")

    mock_store._keyword_retrieval.assert_called_once_with(query="test query", filters={}, top_k=10)
    assert res == {"documents": [doc]}


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._keyword_retrieval_async.return_value = [doc]

    retriever = SupabasePgvectorKeywordRetriever(document_store=mock_store)
    res = await retriever.run_async(query="test query")

    mock_store._keyword_retrieval_async.assert_called_once_with(query="test query", filters={}, top_k=10)
    assert res == {"documents": [doc]}


def test_run_with_filters():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._keyword_retrieval.return_value = [doc]

    init_filter = {"field": "meta.category", "operator": "==", "value": "news"}
    runtime_filter = {"field": "meta.score", "operator": ">", "value": 0.5}
    merged_filter = {"operator": "AND", "conditions": [init_filter, runtime_filter]}

    retriever = SupabasePgvectorKeywordRetriever(
        document_store=mock_store, filter_policy=FilterPolicy.MERGE, filters=init_filter
    )
    res = retriever.run(query="test query", filters=runtime_filter)

    mock_store._keyword_retrieval.assert_called_once_with(query="test query", filters=merged_filter, top_k=10)
    assert res == {"documents": [doc]}


@pytest.mark.asyncio
async def test_run_async_with_filters():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._keyword_retrieval_async.return_value = [doc]

    init_filter = {"field": "meta.category", "operator": "==", "value": "news"}
    runtime_filter = {"field": "meta.score", "operator": ">", "value": 0.5}
    merged_filter = {"operator": "AND", "conditions": [init_filter, runtime_filter]}

    retriever = SupabasePgvectorKeywordRetriever(
        document_store=mock_store, filter_policy=FilterPolicy.MERGE, filters=init_filter
    )
    res = await retriever.run_async(query="test query", filters=runtime_filter)

    mock_store._keyword_retrieval_async.assert_called_once_with(query="test query", filters=merged_filter, top_k=10)
    assert res == {"documents": [doc]}


@pytest.mark.integration
def test_keyword_retrieval(document_store: SupabasePgvectorDocumentStore):
    docs = [
        Document(content="The quick brown fox chased the dog", embedding=[0.1] * 768),
        Document(content="The fox was brown", embedding=[0.1] * 768),
        Document(content="The lazy dog", embedding=[0.1] * 768),
        Document(content="fox fox fox", embedding=[0.1] * 768),
    ]

    document_store.write_documents(docs)

    results = document_store._keyword_retrieval(query="fox", top_k=2)

    assert len(results) == 2
    for doc in results:
        assert "fox" in doc.content
    assert results[0].id == docs[-1].id
    assert results[0].score > results[1].score


@pytest.mark.integration
def test_keyword_retrieval_with_filters(document_store: SupabasePgvectorDocumentStore):
    docs = [
        Document(
            content="The quick brown fox chased the dog",
            embedding=[0.1] * 768,
            meta={"meta_field": "right_value"},
        ),
        Document(content="The fox was brown", embedding=[0.1] * 768, meta={"meta_field": "right_value"}),
        Document(content="The lazy dog", embedding=[0.1] * 768, meta={"meta_field": "right_value"}),
        Document(content="fox fox fox", embedding=[0.1] * 768, meta={"meta_field": "wrong_value"}),
    ]

    document_store.write_documents(docs)

    filters = {"field": "meta.meta_field", "operator": "==", "value": "right_value"}

    results = document_store._keyword_retrieval(query="fox", top_k=3, filters=filters)
    assert len(results) == 2
    for doc in results:
        assert "fox" in doc.content
        assert doc.meta["meta_field"] == "right_value"


@pytest.mark.integration
def test_empty_query(document_store: SupabasePgvectorDocumentStore):
    with pytest.raises(ValueError):
        document_store._keyword_retrieval(query="")
