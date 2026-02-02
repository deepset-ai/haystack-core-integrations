# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSQLRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._raise_on_failure is True


def test_init_custom():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=False, fetch_size=50)
    assert retriever._raise_on_failure is False
    assert retriever._fetch_size == 50


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchSQLRetriever(document_store="not a document store")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchSQLRetriever(document_store=document_store, fetch_size=100)
    res = retriever.to_dict()
    assert (
        res["type"]
        == "haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever"
    )
    assert res["init_parameters"]["raise_on_failure"] is True
    assert res["init_parameters"]["fetch_size"] == 100
    assert "document_store" in res["init_parameters"]


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    data = retriever.to_dict()
    retriever_from_dict = ElasticsearchSQLRetriever.from_dict(data)
    assert retriever_from_dict._raise_on_failure is True


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.return_value = {
        "columns": [{"name": "content", "type": "text"}],
        "rows": [["Test content"]],
    }
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)
    result = retriever.run(query="SELECT content FROM test_index")

    mock_store._query_sql.assert_called_once_with(query="SELECT content FROM test_index", fetch_size=None)
    assert "result" in result
    assert result["result"]["columns"][0]["name"] == "content"
    assert result["result"]["rows"][0][0] == "Test content"


def test_run_with_fetch_size():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.return_value = {"columns": [], "rows": []}
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, fetch_size=10)
    retriever.run(query="SELECT * FROM test_index")

    mock_store._query_sql.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=10)


def test_run_with_runtime_fetch_size():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.return_value = {"columns": [], "rows": []}
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, fetch_size=10)
    retriever.run(query="SELECT * FROM test_index", fetch_size=20)

    mock_store._query_sql.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=20)


def test_run_with_runtime_document_store():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store2 = Mock(spec=ElasticsearchDocumentStore)
    mock_store2._query_sql.return_value = {"columns": [], "rows": []}

    retriever = ElasticsearchSQLRetriever(document_store=mock_store)
    retriever.run(query="SELECT * FROM test_index", document_store=mock_store2)

    mock_store._query_sql.assert_not_called()
    mock_store2._query_sql.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=None)


def test_run_invalid_runtime_document_store():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        retriever.run(query="SELECT * FROM test_index", document_store="invalid")


def test_run_error_handling_raise():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.side_effect = Exception("SQL error")
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=True)

    with pytest.raises(Exception, match="SQL error"):
        retriever.run(query="SELECT * FROM test_index")


def test_run_error_handling_no_raise():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.side_effect = Exception("SQL error")
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=False)

    result = retriever.run(query="SELECT * FROM test_index")
    assert result["result"] == {}


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql_async.return_value = {
        "columns": [{"name": "content", "type": "text"}],
        "rows": [["Test content"]],
    }
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)
    result = await retriever.run_async(query="SELECT content FROM test_index")

    mock_store._query_sql_async.assert_called_once_with(query="SELECT content FROM test_index", fetch_size=None)
    assert "result" in result
    assert result["result"]["columns"][0]["name"] == "content"


@pytest.mark.asyncio
async def test_run_async_with_fetch_size():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql_async.return_value = {"columns": [], "rows": []}
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, fetch_size=10)
    await retriever.run_async(query="SELECT * FROM test_index")

    mock_store._query_sql_async.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=10)


@pytest.mark.asyncio
async def test_run_async_error_handling_raise():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql_async.side_effect = Exception("SQL error")
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=True)

    with pytest.raises(Exception, match="SQL error"):
        await retriever.run_async(query="SELECT * FROM test_index")


@pytest.mark.asyncio
async def test_run_async_error_handling_no_raise():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql_async.side_effect = Exception("SQL error")
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=False)

    result = await retriever.run_async(query="SELECT * FROM test_index")
    assert result["result"] == {}
