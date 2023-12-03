# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from opensearch_haystack.bm25_retriever import OpenSearchBM25Retriever
from opensearch_haystack.document_store import OpenSearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchBM25Retriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert not retriever._scale_score


@patch("opensearch_haystack.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchBM25Retriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "opensearch_haystack.bm25_retriever.OpenSearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                },
                "type": "opensearch_haystack.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": False,
        },
    }


@patch("opensearch_haystack.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    data = {
        "type": "opensearch_haystack.bm25_retriever.OpenSearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "opensearch_haystack.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": True,
        },
    }
    retriever = OpenSearchBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._fuzziness == "AUTO"
    assert retriever._top_k == 10
    assert retriever._scale_score


def test_run():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = OpenSearchBM25Retriever(document_store=mock_store)
    res = retriever.run(query="some query")
    mock_store._bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={},
        fuzziness="AUTO",
        top_k=10,
        scale_score=False,
        all_terms_must_match=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


def test_run_init_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = OpenSearchBM25Retriever(
        document_store=mock_store,
        filters={"from": "init"},
        all_terms_must_match=True,
        scale_score=True,
        top_k=11,
        fuzziness="1",
    )
    res = retriever.run(query="some query")
    mock_store._bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={"from": "init"},
        fuzziness="1",
        top_k=11,
        scale_score=True,
        all_terms_must_match=True,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


def test_run_time_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = OpenSearchBM25Retriever(
        document_store=mock_store,
        filters={"from": "init"},
        all_terms_must_match=True,
        scale_score=True,
        top_k=11,
        fuzziness="1",
    )
    res = retriever.run(
        query="some query",
        filters={"from": "run"},
        all_terms_must_match=False,
        scale_score=False,
        top_k=9,
        fuzziness="2",
    )
    mock_store._bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={"from": "run"},
        fuzziness="2",
        top_k=9,
        scale_score=False,
        all_terms_must_match=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
