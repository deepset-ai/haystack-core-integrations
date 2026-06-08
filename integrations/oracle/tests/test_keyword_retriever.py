# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.oracle import OracleKeywordRetriever


def test_run_calls_keyword_retrieval(mock_store):
    retriever = OracleKeywordRetriever(document_store=mock_store, top_k=5)
    result = retriever.run(query="hello world")
    mock_store._keyword_retrieval.assert_called_once_with("hello world", filters={}, top_k=5)
    assert len(result["documents"]) == 1


def test_run_replace_policy_uses_runtime_filters(mock_store):
    retriever = OracleKeywordRetriever(
        document_store=mock_store,
        filters={"field": "meta.lang", "operator": "==", "value": "en"},
        filter_policy=FilterPolicy.REPLACE,
    )
    runtime_filters = {"field": "meta.year", "operator": ">", "value": 2020}
    retriever.run(query="hello world", filters=runtime_filters)
    call_filters = mock_store._keyword_retrieval.call_args.kwargs["filters"]
    assert call_filters == runtime_filters


def test_run_merge_policy_combines_filters(mock_store):
    retriever = OracleKeywordRetriever(
        document_store=mock_store,
        filters={"field": "meta.lang", "operator": "==", "value": "en"},
        filter_policy=FilterPolicy.MERGE,
    )
    retriever.run(
        query="hello world",
        filters={"field": "meta.year", "operator": ">", "value": 2020},
    )
    call_filters = mock_store._keyword_retrieval.call_args.kwargs["filters"]
    assert call_filters["operator"] == "AND"
    assert len(call_filters["conditions"]) == 2


def test_run_top_k_override(mock_store):
    retriever = OracleKeywordRetriever(document_store=mock_store, top_k=10)
    retriever.run(query="hello world", top_k=3)
    assert mock_store._keyword_retrieval.call_args.kwargs["top_k"] == 3


def test_to_dict_from_dict_roundtrip(mock_store):
    retriever = OracleKeywordRetriever(
        document_store=mock_store,
        top_k=7,
        filters={"field": "meta.x", "operator": "==", "value": "y"},
    )
    d = retriever.to_dict()
    assert d["init_parameters"]["top_k"] == 7
    assert d["init_parameters"]["filters"] == {"field": "meta.x", "operator": "==", "value": "y"}
    assert d["init_parameters"]["filter_policy"] == "replace"

    restored = OracleKeywordRetriever.from_dict(d)
    assert restored.top_k == 7
    assert restored.filters == {"field": "meta.x", "operator": "==", "value": "y"}
    assert restored.filter_policy == FilterPolicy.REPLACE
    assert restored.document_store.table_name == "test_docs"
    assert restored.document_store.embedding_dim == 4


def test_invalid_document_store_raises_type_error():
    with pytest.raises(TypeError, match="must be an instance of OracleDocumentStore"):
        OracleKeywordRetriever(document_store="not_a_store")


@pytest.mark.asyncio
async def test_run_async_calls_async_retrieval(mock_store):
    retriever = OracleKeywordRetriever(document_store=mock_store, top_k=5)
    result = await retriever.run_async(query="hello world")
    mock_store._keyword_retrieval_async.assert_called_once_with("hello world", filters={}, top_k=5)
    assert "documents" in result
