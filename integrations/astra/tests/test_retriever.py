# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.fixture
def mocked_store(monkeypatch):
    monkeypatch.setenv("ASTRA_DB_APPLICATION_TOKEN", "fake-token")
    monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "http://fake-url.apps.astra.datastax.com")
    with patch("haystack_integrations.document_stores.astra.document_store.AstraClient"):
        yield AstraDocumentStore()


def _serialized_retriever(*, include_filter_policy: bool = True) -> dict:
    init_parameters = {
        "filters": {"bar": "baz"},
        "top_k": 42,
        "document_store": {
            "type": "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore",
            "init_parameters": {
                "api_endpoint": {"type": "env_var", "env_vars": ["ASTRA_DB_API_ENDPOINT"], "strict": True},
                "token": {"type": "env_var", "env_vars": ["ASTRA_DB_APPLICATION_TOKEN"], "strict": True},
                "collection_name": "documents",
                "embedding_dimension": 768,
                "duplicates_policy": "NONE",
                "similarity": "cosine",
            },
        },
    }
    if include_filter_policy:
        init_parameters["filter_policy"] = "replace"
    return {
        "type": "haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever",
        "init_parameters": init_parameters,
    }


def test_retriever_init(mocked_store):
    retriever = AstraEmbeddingRetriever(mocked_store, filters={"foo": "bar"}, top_k=99, filter_policy="replace")
    assert retriever.filters == {"foo": "bar"}
    assert retriever.top_k == 99
    assert retriever.document_store == mocked_store
    assert retriever.filter_policy == FilterPolicy.REPLACE

    retriever = AstraEmbeddingRetriever(
        mocked_store, filters={"foo": "bar"}, top_k=99, filter_policy=FilterPolicy.MERGE
    )
    assert retriever.filter_policy == FilterPolicy.MERGE

    with pytest.raises(ValueError):
        AstraEmbeddingRetriever(mocked_store, filters={"foo": "bar"}, top_k=99, filter_policy="unknown")

    with pytest.raises(ValueError):
        AstraEmbeddingRetriever(mocked_store, filters={"foo": "bar"}, top_k=99, filter_policy=None)


def test_retriever_init_rejects_non_astra_document_store():
    with pytest.raises(Exception, match="document_store must be an instance of AstraDocumentStore"):
        AstraEmbeddingRetriever(document_store="not-a-store")  # type: ignore[arg-type]


def test_retriever_to_dict(mocked_store):
    retriever = AstraEmbeddingRetriever(mocked_store, filters={"foo": "bar"}, top_k=99)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever",
        "init_parameters": {
            "filters": {"foo": "bar"},
            "top_k": 99,
            "filter_policy": "replace",
            "document_store": {
                "type": "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore",
                "init_parameters": {
                    "api_endpoint": {"type": "env_var", "env_vars": ["ASTRA_DB_API_ENDPOINT"], "strict": True},
                    "token": {"type": "env_var", "env_vars": ["ASTRA_DB_APPLICATION_TOKEN"], "strict": True},
                    "collection_name": "documents",
                    "embedding_dimension": 768,
                    "duplicates_policy": "NONE",
                    "similarity": "cosine",
                    "namespace": None,
                },
            },
        },
    }


@pytest.mark.parametrize("include_filter_policy", [True, False])
def test_retriever_from_dict(mocked_store, include_filter_policy):  # noqa: ARG001
    retriever = AstraEmbeddingRetriever.from_dict(_serialized_retriever(include_filter_policy=include_filter_policy))
    assert retriever.top_k == 42
    assert retriever.filters == {"bar": "baz"}
    # filter_policy defaults to REPLACE when absent
    assert retriever.filter_policy == FilterPolicy.REPLACE


def test_run_uses_runtime_top_k_and_filters(mocked_store):
    mock_doc = Document(content="test", id="1")
    with patch.object(mocked_store, "search", return_value=[mock_doc]) as mocked_search:
        retriever = AstraEmbeddingRetriever(
            mocked_store, top_k=5, filters={"lang": "en"}, filter_policy=FilterPolicy.REPLACE
        )
        result = retriever.run(query_embedding=[0.1] * 768, filters={"year": 2024}, top_k=3)
        assert result == {"documents": [mock_doc]}
        mocked_search.assert_called_once_with([0.1] * 768, 3, filters={"year": 2024})


@pytest.mark.asyncio
async def test_run_async(mocked_store):
    mock_doc = Document(content="test", id="1")
    with patch.object(mocked_store, "search", return_value=[mock_doc]):
        retriever = AstraEmbeddingRetriever(mocked_store, top_k=5)
        result = await retriever.run_async(query_embedding=[0.1] * 768)
        assert result["documents"] == [mock_doc]
        call_args = mocked_store.search.call_args
        assert call_args.args == ([0.1] * 768, 5)
        assert call_args.kwargs == {"filters": {}}


@pytest.mark.asyncio
async def test_run_async_filters_replace(mocked_store):
    mock_doc = Document(content="test", id="1")
    with patch.object(mocked_store, "search", return_value=[mock_doc]):
        retriever = AstraEmbeddingRetriever(
            mocked_store, top_k=5, filters={"lang": "en"}, filter_policy=FilterPolicy.REPLACE
        )
        await retriever.run_async(query_embedding=[0.1] * 768, filters={"year": 2024})
        assert mocked_store.search.call_args.kwargs["filters"] == {"year": 2024}


@pytest.mark.asyncio
async def test_run_async_filters_merge(mocked_store):
    mock_doc = Document(content="test", id="1")
    init_filters = {"field": "lang", "operator": "==", "value": "en"}
    runtime_filters = {"field": "year", "operator": "==", "value": 2024}
    with patch.object(mocked_store, "search", return_value=[mock_doc]):
        retriever = AstraEmbeddingRetriever(
            mocked_store, top_k=5, filters=init_filters, filter_policy=FilterPolicy.MERGE
        )
        await retriever.run_async(query_embedding=[0.1] * 768, filters=runtime_filters)
        merged = mocked_store.search.call_args.kwargs["filters"]
        assert merged["operator"] == "AND"
        assert init_filters in merged["conditions"]
        assert runtime_filters in merged["conditions"]
