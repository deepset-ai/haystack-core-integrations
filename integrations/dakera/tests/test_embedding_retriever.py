# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.dakera import DakeraEmbeddingRetriever
from haystack_integrations.document_stores.dakera import DakeraDocumentStore


def test_init_default():
    mock_store = Mock(spec=DakeraDocumentStore)
    retriever = DakeraEmbeddingRetriever(document_store=mock_store)
    assert retriever.document_store is mock_store
    assert retriever.filters == {}
    assert retriever.top_k == 10
    assert retriever.filter_policy == FilterPolicy.REPLACE


def test_init_with_parameters():
    mock_store = Mock(spec=DakeraDocumentStore)
    retriever = DakeraEmbeddingRetriever(
        document_store=mock_store,
        filters={"field": "meta.k", "operator": "==", "value": "v"},
        top_k=5,
        filter_policy=FilterPolicy.MERGE,
    )
    assert retriever.top_k == 5
    assert retriever.filter_policy == FilterPolicy.MERGE


def test_init_raises_for_non_dakera_document_store():
    with pytest.raises(ValueError, match="must be an instance of DakeraDocumentStore"):
        DakeraEmbeddingRetriever(document_store="not-a-store")


def test_to_and_from_dict(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "dk-env")
    document_store = DakeraDocumentStore(namespace="docs", dimension=384)
    retriever = DakeraEmbeddingRetriever(document_store=document_store, top_k=7)

    data = retriever.to_dict()
    assert data["type"].endswith("DakeraEmbeddingRetriever")
    assert data["init_parameters"]["top_k"] == 7
    assert data["init_parameters"]["filter_policy"] == "replace"
    assert data["init_parameters"]["document_store"]["init_parameters"]["namespace"] == "docs"

    restored = DakeraEmbeddingRetriever.from_dict(data)
    assert isinstance(restored.document_store, DakeraDocumentStore)
    assert restored.top_k == 7
    assert restored.filter_policy == FilterPolicy.REPLACE


def test_from_dict_no_filter_policy(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "dk-env")
    document_store = DakeraDocumentStore(namespace="docs", dimension=384)
    data = DakeraEmbeddingRetriever(document_store=document_store).to_dict()
    del data["init_parameters"]["filter_policy"]

    restored = DakeraEmbeddingRetriever.from_dict(data)
    # Missing filter_policy defaults to REPLACE.
    assert restored.filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=DakeraDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="alpha")]
    retriever = DakeraEmbeddingRetriever(document_store=mock_store, top_k=3)

    result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

    mock_store._embedding_retrieval.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], filters={}, top_k=3)
    assert result["documents"][0].content == "alpha"


def test_run_overrides_top_k_and_filters():
    mock_store = Mock(spec=DakeraDocumentStore)
    mock_store._embedding_retrieval.return_value = []
    retriever = DakeraEmbeddingRetriever(document_store=mock_store, top_k=3)

    runtime_filters = {"field": "meta.k", "operator": "==", "value": "v"}
    retriever.run(query_embedding=[0.1], filters=runtime_filters, top_k=9)

    mock_store._embedding_retrieval.assert_called_once_with(query_embedding=[0.1], filters=runtime_filters, top_k=9)


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=DakeraDocumentStore)

    async def fake_retrieval(*, query_embedding, filters, top_k):  # noqa: ARG001
        return [Document(content="beta")]

    mock_store._embedding_retrieval_async = fake_retrieval
    retriever = DakeraEmbeddingRetriever(document_store=mock_store, top_k=3)

    result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3])
    assert result["documents"][0].content == "beta"
