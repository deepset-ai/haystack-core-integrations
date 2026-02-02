# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSQLRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


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
async def test_run_async_with_runtime_fetch_size():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql_async.return_value = {"columns": [], "rows": []}
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, fetch_size=10)
    await retriever.run_async(query="SELECT * FROM test_index", fetch_size=20)

    mock_store._query_sql_async.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=20)


@pytest.mark.asyncio
async def test_run_async_invalid_runtime_document_store():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        await retriever.run_async(query="SELECT * FROM test_index", document_store="invalid")


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_basic_query(document_store: ElasticsearchDocumentStore):
    """Test basic async SQL query execution"""
    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
    ]
    await document_store.write_documents_async(docs, refresh=True)

    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    sql_query = (
        f'SELECT content, category, status FROM "{document_store._index}" '  # noqa: S608
        f"WHERE category = 'A' ORDER BY priority"
    )
    result = await retriever.run_async(query=sql_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)
    assert "columns" in response
    assert "rows" in response
    assert len(response["rows"]) == 2

    column_names = [c["name"] for c in response["columns"]]
    category_idx = column_names.index("category")
    for row in response["rows"]:
        assert row[category_idx] == "A"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_runtime_document_store_switching(
    document_store: ElasticsearchDocumentStore, document_store_2: ElasticsearchDocumentStore
):
    """Test async switching document stores at runtime"""
    docs1 = [
        Document(content="Python programming", meta={"category": "A"}),
        Document(content="Java programming", meta={"category": "B"}),
    ]
    await document_store.write_documents_async(docs1, refresh=True)

    docs2 = [
        Document(content="JavaScript development", meta={"category": "C"}),
        Document(content="TypeScript development", meta={"category": "D"}),
    ]
    await document_store_2.write_documents_async(docs2, refresh=True)

    retriever = ElasticsearchSQLRetriever(document_store=document_store)

    sql_query1 = f"SELECT content, category FROM \"{document_store._index}\" WHERE category = 'A'"  # noqa: S608
    result1 = await retriever.run_async(query=sql_query1)
    assert len(result1["result"]["rows"]) == 1
    content_col_idx = next(i for i, c in enumerate(result1["result"]["columns"]) if c["name"] == "content")
    assert "Python" in result1["result"]["rows"][0][content_col_idx]

    sql_query2 = f"SELECT content, category FROM \"{document_store_2._index}\" WHERE category = 'C'"  # noqa: S608
    result2 = await retriever.run_async(query=sql_query2, document_store=document_store_2)
    assert len(result2["result"]["rows"]) == 1
    assert "JavaScript" in result2["result"]["rows"][0][content_col_idx]

    assert result1["result"]["rows"][0][content_col_idx] != result2["result"]["rows"][0][content_col_idx]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_error_handling(document_store: ElasticsearchDocumentStore):
    """Test async error handling for invalid SQL queries"""
    retriever = ElasticsearchSQLRetriever(document_store=document_store, raise_on_failure=True)

    invalid_query = "SELECT * FROM non_existent_index"
    with pytest.raises(Exception, match="Failed to execute SQL query"):
        await retriever.run_async(query=invalid_query)

    retriever_no_raise = ElasticsearchSQLRetriever(document_store=document_store, raise_on_failure=False)
    result = await retriever_no_raise.run_async(query=invalid_query)
    assert result["result"] == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_with_fetch_size(document_store: ElasticsearchDocumentStore):
    """Test async SQL retriever with fetch_size parameter"""
    docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
    await document_store.write_documents_async(docs, refresh=True)

    retriever = ElasticsearchSQLRetriever(document_store=document_store, fetch_size=5)
    sql_query = (
        f'SELECT content, category, "index" FROM "{document_store._index}" '  # noqa: S608
        f"WHERE category = 'A' ORDER BY \"index\""
    )

    result = await retriever.run_async(query=sql_query)
    assert "result" in result
    assert isinstance(result["result"], dict)
    assert "columns" in result["result"]
    assert "rows" in result["result"]
    assert len(result["result"]["rows"]) > 0

    result2 = await retriever.run_async(query=sql_query, fetch_size=10)
    assert "result" in result2
    assert isinstance(result2["result"], dict)
    assert "rows" in result2["result"]
    assert len(result2["result"]["rows"]) > 0
