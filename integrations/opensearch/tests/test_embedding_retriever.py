# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.document_store import DEFAULT_MAX_CHUNK_BYTES


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert retriever._efficient_filtering is False

    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        OpenSearchEmbeddingRetriever(document_store=mock_store, filter_policy="unknown")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchEmbeddingRetriever(document_store=document_store, custom_query={"some": "custom query"})
    res = retriever.to_dict()
    type_s = "haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever"
    assert res == {
        "type": type_s,
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "embedding_dim": 768,
                    "hosts": "some fake host",
                    "index": "default",
                    "mappings": {
                        "dynamic_templates": [
                            {
                                "strings": {
                                    "mapping": {
                                        "type": "keyword",
                                    },
                                    "match_mapping_type": "string",
                                },
                            },
                        ],
                        "properties": {
                            "content": {
                                "type": "text",
                            },
                            "embedding": {
                                "dimension": 768,
                                "index": True,
                                "type": "knn_vector",
                            },
                        },
                    },
                    "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                    "method": None,
                    "settings": {
                        "index.knn": True,
                    },
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
            "top_k": 10,
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": True,
            "efficient_filtering": False,
        },
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    type_s = "haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever"
    data = {
        "type": type_s,
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
            "custom_query": {"some": "custom query"},
            "raise_on_failure": False,
            "efficient_filtering": True,
        },
    }
    retriever = OpenSearchEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._custom_query == {"some": "custom query"}
    assert retriever._raise_on_failure is False
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert retriever._efficient_filtering is True

    # For backwards compatibility with older versions of the retriever without a filter policy
    data = {
        "type": type_s,
        "init_parameters": {
            "document_store": {
                "init_parameters": {"hosts": "some fake host", "index": "default"},
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "custom_query": {"some": "custom query"},
            "raise_on_failure": False,
        },
    }
    retriever = OpenSearchEmbeddingRetriever.from_dict(data)
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        custom_query=None,
        efficient_filtering=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    res = await retriever.run_async(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        custom_query=None,
        efficient_filtering=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_init_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(
        document_store=mock_store,
        filters={"from": "init"},
        top_k=11,
        custom_query="custom_query",
        efficient_filtering=True,
    )
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "init"},
        top_k=11,
        custom_query="custom_query",
        efficient_filtering=True,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_async_init_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(
        document_store=mock_store,
        filters={"from": "init"},
        top_k=11,
        custom_query="custom_query",
    )
    res = await retriever.run_async(query_embedding=[0.5, 0.7])
    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "init"},
        top_k=11,
        custom_query="custom_query",
        efficient_filtering=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_time_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, filters={"from": "init"}, top_k=11)
    res = retriever.run(query_embedding=[0.5, 0.7], filters={"from": "run"}, top_k=9, efficient_filtering=True)
    mock_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "run"},
        top_k=9,
        custom_query=None,
        efficient_filtering=True,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_async_time_params():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval_async.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, filters={"from": "init"}, top_k=11)
    res = await retriever.run_async(query_embedding=[0.5, 0.7], filters={"from": "run"}, top_k=9)
    mock_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={"from": "run"},
        top_k=9,
        custom_query=None,
        efficient_filtering=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
    assert res["documents"][0].embedding == [0.1, 0.2]


def test_run_ignore_errors(caplog):
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.side_effect = Exception("Some error")
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store, raise_on_failure=False)
    res = retriever.run(query_embedding=[0.5, 0.7])
    assert len(res) == 1
    assert res["documents"] == []
    assert "Some error" in caplog.text


def test_run_with_runtime_document_store():
    """Test that runtime document store switching works correctly."""
    # Setup initial document store
    initial_store = Mock(spec=OpenSearchDocumentStore)
    initial_store._embedding_retrieval.return_value = [Document(content="Initial store doc", embedding=[0.1, 0.2])]

    # Setup runtime document store
    runtime_store = Mock(spec=OpenSearchDocumentStore)
    runtime_store._embedding_retrieval.return_value = [Document(content="Runtime store doc", embedding=[0.3, 0.4])]

    retriever = OpenSearchEmbeddingRetriever(document_store=initial_store)

    # Test with runtime document store
    res = retriever.run(query_embedding=[0.5, 0.7], document_store=runtime_store)

    # Verify runtime store was called, not initial store
    runtime_store._embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        custom_query=None,
        efficient_filtering=False,
    )
    initial_store._embedding_retrieval.assert_not_called()

    # Verify results
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Runtime store doc"
    assert res["documents"][0].embedding == [0.3, 0.4]


@pytest.mark.asyncio
async def test_run_async_with_runtime_document_store():
    """Test that runtime document store switching works correctly in async mode."""
    # Setup initial document store
    initial_store = Mock(spec=OpenSearchDocumentStore)
    initial_store._embedding_retrieval_async.return_value = [
        Document(content="Initial store doc", embedding=[0.1, 0.2])
    ]

    # Setup runtime document store
    runtime_store = Mock(spec=OpenSearchDocumentStore)
    runtime_store._embedding_retrieval_async.return_value = [
        Document(content="Runtime store doc", embedding=[0.3, 0.4])
    ]

    retriever = OpenSearchEmbeddingRetriever(document_store=initial_store)

    # Test with runtime document store
    res = await retriever.run_async(query_embedding=[0.5, 0.7], document_store=runtime_store)

    # Verify runtime store was called, not initial store
    runtime_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        custom_query=None,
        efficient_filtering=False,
    )
    initial_store._embedding_retrieval_async.assert_not_called()

    # Verify results
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Runtime store doc"
    assert res["documents"][0].embedding == [0.3, 0.4]


@pytest.mark.integration
def test_embedding_retriever_runtime_document_store_switching(
    document_store, document_store_2, test_documents_with_embeddings_1, test_documents_with_embeddings_2
):
    document_store.write_documents(test_documents_with_embeddings_1)
    document_store_2.write_documents(test_documents_with_embeddings_2)
    retriever = OpenSearchEmbeddingRetriever(document_store=document_store)

    # query embedding to match functional programming languages
    query_embedding = [0.2, 0.3, 0.4] + [0.0] * 765
    results_1 = retriever.run(query_embedding=query_embedding)
    assert len(results_1["documents"]) > 0
    assert any("Haskell" in doc.content for doc in results_1["documents"])
    assert any("Lisp" in doc.content for doc in results_1["documents"])

    # opensearch-2 at runtime with query embedding to match search-related documents
    search_query_embedding = [0.8, 0.7, 0.6] + [0.0] * 765
    results_2 = retriever.run(query_embedding=search_query_embedding, document_store=document_store_2)
    assert len(results_2["documents"]) == 3
    assert any("OpenSearch" in doc.content for doc in results_2["documents"])
    assert any("Elasticsearch" in doc.content for doc in results_2["documents"])
    assert any("Vector databases" in doc.content for doc in results_2["documents"])

    # verify results are different
    store_1_content = [doc.content for doc in results_1["documents"]]
    store_2_content = [doc.content for doc in results_2["documents"]]
    assert not any(content in store_2_content for content in store_1_content)
    assert not any(content in store_1_content for content in store_2_content)

    # opensearch-1 again to ensure we can switch back
    python_query_embedding = [0.4, 0.4, 0.4] + [0.0] * 765
    results_1_again = retriever.run(query_embedding=python_query_embedding)
    assert "Python" in results_1_again["documents"][0].content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_retriever_runtime_document_store_switching_async(
    document_store, document_store_2, test_documents_with_embeddings_1, test_documents_with_embeddings_2
):
    document_store.write_documents(test_documents_with_embeddings_1)
    document_store_2.write_documents(test_documents_with_embeddings_2)
    retriever = OpenSearchEmbeddingRetriever(document_store=document_store)

    # query embedding to match functional programming languages
    query_embedding = [0.2, 0.3, 0.4] + [0.0] * 765
    results_1 = await retriever.run_async(query_embedding=query_embedding)
    assert len(results_1["documents"]) > 0
    assert any("Haskell" in doc.content for doc in results_1["documents"])
    assert any("Lisp" in doc.content for doc in results_1["documents"])

    # opensearch-2 at runtime with query embedding to match search-related documents
    search_query_embedding = [0.8, 0.7, 0.6] + [0.0] * 765
    results_2 = await retriever.run_async(query_embedding=search_query_embedding, document_store=document_store_2)
    assert len(results_2["documents"]) == 3
    assert any("OpenSearch" in doc.content for doc in results_2["documents"])
    assert any("Elasticsearch" in doc.content for doc in results_2["documents"])
    assert any("Vector databases" in doc.content for doc in results_2["documents"])

    # verify results are different
    store_1_content = [doc.content for doc in results_1["documents"]]
    store_2_content = [doc.content for doc in results_2["documents"]]
    assert not any(content in store_2_content for content in store_1_content)
    assert not any(content in store_1_content for content in store_2_content)

    # opensearch-1 again to ensure we can switch back
    python_query_embedding = [0.4, 0.4, 0.4] + [0.0] * 765
    results_1_again = await retriever.run_async(query_embedding=python_query_embedding)
    assert "Python" in results_1_again["documents"][0].content
