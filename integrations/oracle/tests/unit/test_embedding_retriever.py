from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever
from haystack_integrations.document_stores.oracle import OracleDocumentStore


@pytest.fixture()
def mock_store():
    store = MagicMock(spec=OracleDocumentStore)
    store.distance_metric = "COSINE"
    store._embedding_retrieval.return_value = [Document(id="A" * 32, content="hi")]
    store._embedding_retrieval_async.return_value = [Document(id="A" * 32, content="hi")]
    store.to_dict.return_value = {
        "type": "haystack_integrations.document_stores.oracle.document_store.OracleDocumentStore",
        "init_parameters": {
            "connection_config": {
                "user": "u",
                "password": {"type": "token", "token": "p"},
                "dsn": "localhost/xe",
                "wallet_location": None,
                "wallet_password": None,
                "min_connections": 1,
                "max_connections": 5,
            },
            "table_name": "test_docs",
            "embedding_dim": 4,
            "distance_metric": "COSINE",
            "create_table_if_not_exists": False,
            "create_index": False,
            "hnsw_neighbors": 32,
            "hnsw_ef_construction": 200,
            "hnsw_accuracy": 95,
            "hnsw_parallel": 4,
        },
    }
    return store


def test_run_calls_embedding_retrieval(mock_store):
    retriever = OracleEmbeddingRetriever(document_store=mock_store, top_k=5)
    result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])
    mock_store._embedding_retrieval.assert_called_once_with([0.1, 0.2, 0.3, 0.4], filters=None, top_k=5)
    assert len(result["documents"]) == 1


def test_run_merges_filters(mock_store):
    retriever = OracleEmbeddingRetriever(
        document_store=mock_store,
        filters={"field": "meta.lang", "operator": "==", "value": "en"},
    )
    retriever.run(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        filters={"field": "meta.year", "operator": ">", "value": 2020},
    )
    call_filters = mock_store._embedding_retrieval.call_args.kwargs["filters"]
    assert call_filters["operator"] == "AND"
    assert len(call_filters["conditions"]) == 2


def test_run_top_k_override(mock_store):
    retriever = OracleEmbeddingRetriever(document_store=mock_store, top_k=10)
    retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=3)
    assert mock_store._embedding_retrieval.call_args.kwargs["top_k"] == 3


def test_to_dict_from_dict_roundtrip(mock_store, monkeypatch):
    retriever = OracleEmbeddingRetriever(
        document_store=mock_store,
        top_k=7,
        filters={"field": "meta.x", "operator": "==", "value": "y"},
    )
    d = retriever.to_dict()
    assert d["init_parameters"]["top_k"] == 7
    assert d["init_parameters"]["filters"] == {"field": "meta.x", "operator": "==", "value": "y"}


@pytest.mark.asyncio
async def test_run_async_calls_async_retrieval(mock_store):
    retriever = OracleEmbeddingRetriever(document_store=mock_store, top_k=5)
    result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4])
    mock_store._embedding_retrieval_async.assert_called_once()
    assert "documents" in result
