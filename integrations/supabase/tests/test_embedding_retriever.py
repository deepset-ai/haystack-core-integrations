# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils.auth import EnvVarSecret
from numpy.random import rand

from haystack_integrations.components.retrievers.supabase import SupabasePgvectorEmbeddingRetriever
from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore


def test_init_default(mock_store):
    retriever = SupabasePgvectorEmbeddingRetriever(document_store=mock_store)
    assert retriever.document_store == mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE
    assert retriever.vector_function == mock_store.vector_function


def test_init_filter_policy_string(mock_store):
    retriever = SupabasePgvectorEmbeddingRetriever(document_store=mock_store, filter_policy="merge")
    assert retriever.filter_policy == FilterPolicy.MERGE


def test_init_invalid_filter_policy(mock_store):
    with pytest.raises(ValueError):
        SupabasePgvectorEmbeddingRetriever(document_store=mock_store, filter_policy="invalid")


def test_init(mock_store):
    retriever = SupabasePgvectorEmbeddingRetriever(
        document_store=mock_store, filters={"field": "value"}, top_k=5, vector_function="l2_distance"
    )
    assert retriever.document_store == mock_store
    assert retriever.filters == {"field": "value"}
    assert retriever.top_k == 5
    assert retriever.filter_policy == FilterPolicy.REPLACE
    assert retriever.vector_function == "l2_distance"


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="must be an instance of SupabasePgvectorDocumentStore"):
        SupabasePgvectorEmbeddingRetriever(document_store="not a store")


def test_to_dict(mock_store):
    retriever = SupabasePgvectorEmbeddingRetriever(
        document_store=mock_store, filters={"field": "value"}, top_k=5, vector_function="l2_distance"
    )
    res = retriever.to_dict()
    t = "haystack_integrations.components.retrievers.supabase.embedding_retriever.SupabasePgvectorEmbeddingRetriever"
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
            "vector_function": "l2_distance",
            "filter_policy": "replace",
        },
    }


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_from_dict(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some-connection-string")
    t = "haystack_integrations.components.retrievers.supabase.embedding_retriever.SupabasePgvectorEmbeddingRetriever"
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
            "vector_function": "l2_distance",
            "filter_policy": "replace",
        },
    }

    retriever = SupabasePgvectorEmbeddingRetriever.from_dict(data)
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
    assert retriever.vector_function == "l2_distance"


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_from_dict_without_filter_policy(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some-connection-string")
    t = "haystack_integrations.components.retrievers.supabase.embedding_retriever.SupabasePgvectorEmbeddingRetriever"
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
            "vector_function": "l2_distance",
        },
    }

    retriever = SupabasePgvectorEmbeddingRetriever.from_dict(data)
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
    assert retriever.top_k == 5
    assert retriever.filter_policy == FilterPolicy.REPLACE  # defaults to REPLACE
    assert retriever.vector_function == "l2_distance"


def test_run():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    mock_store.vector_function = "cosine_similarity"
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._embedding_retrieval.return_value = [doc]

    retriever = SupabasePgvectorEmbeddingRetriever(document_store=mock_store, vector_function="l2_distance")
    res = retriever.run(query_embedding=[0.3, 0.5])

    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.3, 0.5], filters={}, top_k=10, vector_function="l2_distance"
    )
    assert res == {"documents": [doc]}


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    mock_store.vector_function = "cosine_similarity"
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._embedding_retrieval_async.return_value = [doc]

    retriever = SupabasePgvectorEmbeddingRetriever(document_store=mock_store, vector_function="l2_distance")
    res = await retriever.run_async(query_embedding=[0.3, 0.5])

    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.3, 0.5], filters={}, top_k=10, vector_function="l2_distance"
    )
    assert res == {"documents": [doc]}


def test_run_with_filters():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    mock_store.vector_function = "cosine_similarity"
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._embedding_retrieval.return_value = [doc]

    init_filter = {"field": "meta.category", "operator": "==", "value": "news"}
    runtime_filter = {"field": "meta.score", "operator": ">", "value": 0.5}
    merged_filter = {"operator": "AND", "conditions": [init_filter, runtime_filter]}

    retriever = SupabasePgvectorEmbeddingRetriever(
        document_store=mock_store, filter_policy=FilterPolicy.MERGE, filters=init_filter, vector_function="l2_distance"
    )
    res = retriever.run(query_embedding=[0.3, 0.5], filters=runtime_filter)

    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.3, 0.5], filters=merged_filter, top_k=10, vector_function="l2_distance"
    )
    assert res == {"documents": [doc]}


@pytest.mark.asyncio
async def test_run_async_with_filters():
    mock_store = Mock(spec=SupabasePgvectorDocumentStore)
    mock_store.vector_function = "cosine_similarity"
    doc = Document(content="Test doc", embedding=[0.1, 0.2])
    mock_store._embedding_retrieval_async.return_value = [doc]

    init_filter = {"field": "meta.category", "operator": "==", "value": "news"}
    runtime_filter = {"field": "meta.score", "operator": ">", "value": 0.5}
    merged_filter = {"operator": "AND", "conditions": [init_filter, runtime_filter]}

    retriever = SupabasePgvectorEmbeddingRetriever(
        document_store=mock_store, filter_policy=FilterPolicy.MERGE, filters=init_filter, vector_function="l2_distance"
    )
    res = await retriever.run_async(query_embedding=[0.3, 0.5], filters=runtime_filter)

    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.3, 0.5], filters=merged_filter, top_k=10, vector_function="l2_distance"
    )
    assert res == {"documents": [doc]}


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
def test_embedding_retrieval_cosine_similarity(document_store: SupabasePgvectorDocumentStore):
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


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
def test_embedding_retrieval_inner_product(document_store: SupabasePgvectorDocumentStore):
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


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
def test_embedding_retrieval_l2_distance(document_store: SupabasePgvectorDocumentStore):
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


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
def test_embedding_retrieval_with_filters(document_store: SupabasePgvectorDocumentStore):
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


@pytest.mark.integration
def test_empty_query_embedding(document_store: SupabasePgvectorDocumentStore):
    query_embedding: list[float] = []
    with pytest.raises(ValueError):
        document_store._embedding_retrieval(query_embedding=query_embedding)


@pytest.mark.integration
def test_query_embedding_wrong_dimension(document_store: SupabasePgvectorDocumentStore):
    query_embedding = [0.1] * 4
    with pytest.raises(ValueError):
        document_store._embedding_retrieval(query_embedding=query_embedding)
