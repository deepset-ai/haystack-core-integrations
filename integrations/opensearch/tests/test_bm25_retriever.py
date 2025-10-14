# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

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
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = OpenSearchBM25Retriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        OpenSearchBM25Retriever(document_store=mock_store, filter_policy="unknown")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchBM25Retriever(document_store=document_store, custom_query={"some": "custom query"})
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
                    "http_auth": None,
                    "use_ssl": None,
                    "verify_certs": None,
                    "timeout": None,
                },
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "fuzziness": "AUTO",
            "top_k": 10,
            "scale_score": False,
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": True,
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
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": False,
        },
    }
    retriever = OpenSearchBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._fuzziness == "AUTO"
    assert retriever._top_k == 10
    assert retriever._scale_score
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert retriever._custom_query == {"some": "custom query"}
    assert retriever._raise_on_failure is False

    # For backwards compatibility with older versions of the retriever without a filter policy
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
            "custom_query": {"some": "custom query"},
            "raise_on_failure": False,
        },
    }
    retriever = OpenSearchBM25Retriever.from_dict(data)
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict_not_defaults(_mock_opensearch_client):
    data = {
        "type": "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "fuzziness": 0,
            "top_k": 15,
            "scale_score": True,
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": True,
        },
    }
    retriever = OpenSearchBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._fuzziness == 0
    assert retriever._top_k == 15
    assert retriever._scale_score
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert retriever._custom_query == {"some": "custom query"}
    assert retriever._raise_on_failure is True


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


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval_async.return_value = [Document(content="Test doc")]
    retriever = OpenSearchBM25Retriever(document_store=mock_store)
    res = await retriever.run_async(query="some query")
    mock_store._bm25_retrieval_async.assert_called_once_with(
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


@pytest.mark.asyncio
async def test_run_init_params_async():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval_async.return_value = [Document(content="Test doc")]
    retriever = OpenSearchBM25Retriever(
        document_store=mock_store,
        filters={"from": "init"},
        all_terms_must_match=True,
        scale_score=True,
        top_k=11,
        fuzziness="1",
        custom_query={"some": "custom query"},
    )
    res = await retriever.run_async(query="some query")
    mock_store._bm25_retrieval_async.assert_called_once_with(
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


@pytest.mark.asyncio
async def test_run_time_params_async():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval_async.return_value = [Document(content="Test doc")]
    retriever = OpenSearchBM25Retriever(
        document_store=mock_store,
        filters={"from": "init"},
        all_terms_must_match=True,
        scale_score=True,
        top_k=11,
        fuzziness="1",
    )
    res = await retriever.run_async(
        query="some query",
        filters={"from": "run"},
        all_terms_must_match=False,
        scale_score=False,
        top_k=9,
        fuzziness="2",
    )
    mock_store._bm25_retrieval_async.assert_called_once_with(
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


def test_run_ignore_errors(caplog):
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval.side_effect = Exception("Some error")
    retriever = OpenSearchBM25Retriever(document_store=mock_store, raise_on_failure=False)
    res = retriever.run(query="some query")
    assert len(res) == 1
    assert res["documents"] == []
    assert "Some error" in caplog.text


def test_run_with_runtime_document_store():
    # initial document store
    initial_store = Mock(spec=OpenSearchDocumentStore)
    initial_store._bm25_retrieval.return_value = [Document(content="Initial store doc")]

    # runtime document store
    runtime_store = Mock(spec=OpenSearchDocumentStore)
    runtime_store._bm25_retrieval.return_value = [Document(content="Runtime store doc")]

    retriever = OpenSearchBM25Retriever(document_store=initial_store)

    res = retriever.run(query="some query", document_store=runtime_store)

    # verify runtime store was called, not initial store
    runtime_store._bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={},
        fuzziness="AUTO",
        top_k=10,
        scale_score=False,
        all_terms_must_match=False,
        custom_query=None,
    )
    initial_store._bm25_retrieval.assert_not_called()

    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Runtime store doc"


@pytest.mark.asyncio
async def test_run_async_with_runtime_document_store():
    # initial document store
    initial_store = Mock(spec=OpenSearchDocumentStore)
    initial_store._bm25_retrieval_async.return_value = [Document(content="Initial store doc")]

    # runtime document store
    runtime_store = Mock(spec=OpenSearchDocumentStore)
    runtime_store._bm25_retrieval_async.return_value = [Document(content="Runtime store doc")]

    retriever = OpenSearchBM25Retriever(document_store=initial_store)

    res = await retriever.run_async(query="some query", document_store=runtime_store)

    # verify runtime store was called, not initial store
    runtime_store._bm25_retrieval_async.assert_called_once_with(
        query="some query",
        filters={},
        fuzziness="AUTO",
        top_k=10,
        scale_score=False,
        all_terms_must_match=False,
        custom_query=None,
    )
    initial_store._bm25_retrieval_async.assert_not_called()

    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Runtime store doc"


@pytest.mark.integration
def test_bm25_retriever_runtime_document_store_switching(
    document_store, document_store_2, test_documents_with_embeddings_1, test_documents_with_embeddings_2
):
    # Write documents to opensearch-1
    document_store.write_documents(test_documents_with_embeddings_1)

    # Write documents to opensearch-2
    document_store_2.write_documents(test_documents_with_embeddings_2)

    # Initialize BM25 retriever with opensearch-1
    retriever = OpenSearchBM25Retriever(document_store=document_store)

    # only results from opensearch-1
    results_1 = retriever.run(query="programming language")
    assert any("Haskell" in doc.content for doc in results_1["documents"])
    assert any("Lisp" in doc.content for doc in results_1["documents"])

    # switch to opensearch-2 at runtime
    results_2 = retriever.run(query="search", document_store=document_store_2)

    # only results from opensearch-2
    assert len(results_2["documents"]) == 3
    assert any("OpenSearch" in doc.content for doc in results_2["documents"])
    assert any("Elasticsearch" in doc.content for doc in results_2["documents"])

    # Test 3: Verify the results are different (proving we're querying different stores)
    store_1_content = [doc.content for doc in results_1["documents"]]
    store_2_content = [doc.content for doc in results_2["documents"]]

    # The content should be completely different
    assert not any(content in store_2_content for content in store_1_content)
    assert not any(content in store_1_content for content in store_2_content)

    # Test 4: Query opensearch-1 again to ensure we can switch back
    results_1_again = retriever.run(query="Python")

    # Should get results from opensearch-1 (the initial store)
    assert len(results_1_again["documents"]) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bm25_retriever_async_runtime_document_store_switching(
    document_store, document_store_2, test_documents_with_embeddings_1, test_documents_with_embeddings_2
):
    # Write documents to opensearch-1
    document_store.write_documents(test_documents_with_embeddings_1)

    # Write documents to opensearch-2
    document_store_2.write_documents(test_documents_with_embeddings_2)

    # Initialize BM25 retriever with opensearch-1
    retriever = OpenSearchBM25Retriever(document_store=document_store)

    # only results from opensearch-1
    results_1 = await retriever.run_async(query="programming language")
    assert any("Haskell" in doc.content for doc in results_1["documents"])
    assert any("Lisp" in doc.content for doc in results_1["documents"])

    # switch to opensearch-2 at runtime
    results_2 = await retriever.run_async(query="search", document_store=document_store_2)

    # only results from opensearch-2
    assert len(results_2["documents"]) == 3
    assert any("OpenSearch" in doc.content for doc in results_2["documents"])
    assert any("Elasticsearch" in doc.content for doc in results_2["documents"])

    # Test 3: Verify the results are different (proving we're querying different stores)
    store_1_content = [doc.content for doc in results_1["documents"]]
    store_2_content = [doc.content for doc in results_2["documents"]]

    # The content should be completely different
    assert not any(content in store_2_content for content in store_1_content)
    assert not any(content in store_1_content for content in store_2_content)

    # Test 4: Query opensearch-1 again to ensure we can switch back
    results_1_again = await retriever.run_async(query="Python")

    # Should get results from opensearch-1 (the initial store)
    assert len(results_1_again["documents"]) == 1
