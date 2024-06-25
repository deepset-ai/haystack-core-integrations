# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document
from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.document_store import DEFAULT_MAX_CHUNK_BYTES


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchBM25Retriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert not retriever._scale_score


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchBM25Retriever(document_store=document_store)
    res = retriever.to_dict()
    assert res == {
        "type": "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "embedding_dim": 768,
                    "hosts": "some fake host",
                    "index": "default",
                    "mappings": {
                        "dynamic_templates": [
                            {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                        ],
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                        },
                    },
                    "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                    "method": None,
                    "settings": {"index.knn": True},
                    "return_embedding": False,
                    "create_index": True,
                },
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": False,
        },
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    data = {
        "type": "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
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
        custom_query=None,
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
        custom_query={"some": "custom query"},
    )
    res = retriever.run(query="some query")
    mock_store._bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={"from": "init"},
        fuzziness="1",
        top_k=11,
        scale_score=True,
        all_terms_must_match=True,
        custom_query={"some": "custom query"},
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
        custom_query=None,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
