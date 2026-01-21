# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.opensearch import OpenSearchSQLRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchSQLRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._raise_on_failure is True


def test_init_custom():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchSQLRetriever(document_store=mock_store, raise_on_failure=False, fetch_size=50)
    assert retriever._raise_on_failure is False
    assert retriever._fetch_size == 50


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="document_store must be an instance of OpenSearchDocumentStore"):
        OpenSearchSQLRetriever(document_store="not a document store")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchSQLRetriever(document_store=document_store, fetch_size=100)
    res = retriever.to_dict()
    assert res["type"] == "haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever"
    assert res["init_parameters"]["raise_on_failure"] is True
    assert res["init_parameters"]["fetch_size"] == 100
    assert "document_store" in res["init_parameters"]


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchSQLRetriever(document_store=document_store)
    data = retriever.to_dict()
    retriever_from_dict = OpenSearchSQLRetriever.from_dict(data)
    assert retriever_from_dict._raise_on_failure is True


def test_run():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql.return_value = [{"content": "Test doc", "category": "A"}]
    retriever = OpenSearchSQLRetriever(document_store=mock_store)
    res = retriever.run(query="SELECT content, category FROM my_index WHERE category = 'A'")
    mock_store._query_sql.assert_called_once_with(
        query="SELECT content, category FROM my_index WHERE category = 'A'",
        cursor=None,
        fetch_size=None,
    )
    assert len(res) == 1
    assert "result" in res
    assert res["result"] == [{"content": "Test doc", "category": "A"}]


def test_run_with_runtime_document_store():
    mock_store1 = Mock(spec=OpenSearchDocumentStore)
    mock_store2 = Mock(spec=OpenSearchDocumentStore)
    mock_store2._query_sql.return_value = [{"result": "from store 2"}]
    retriever = OpenSearchSQLRetriever(document_store=mock_store1)
    res = retriever.run(query="SELECT * FROM my_index", document_store=mock_store2)
    mock_store1._query_sql.assert_not_called()
    mock_store2._query_sql.assert_called_once_with(query="SELECT * FROM my_index", cursor=None, fetch_size=None)
    assert res["result"] == [{"result": "from store 2"}]


def test_run_with_cursor_and_fetch_size():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql.return_value = [{"content": "Test doc", "category": "A"}]
    retriever = OpenSearchSQLRetriever(document_store=mock_store, fetch_size=50)
    res = retriever.run(query="SELECT * FROM my_index", cursor="test_cursor", fetch_size=100)
    mock_store._query_sql.assert_called_once_with(
        query="SELECT * FROM my_index",
        cursor="test_cursor",
        fetch_size=100,
    )
    assert res["result"] == [{"content": "Test doc", "category": "A"}]


def test_run_with_error_raise_on_failure():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql.side_effect = Exception("SQL error")
    retriever = OpenSearchSQLRetriever(document_store=mock_store, raise_on_failure=True)
    with pytest.raises(Exception, match="SQL error"):
        retriever.run(query="SELECT * FROM my_index")


def test_run_with_error_no_raise():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql.side_effect = Exception("SQL error")
    retriever = OpenSearchSQLRetriever(document_store=mock_store, raise_on_failure=False)
    res = retriever.run(query="SELECT * FROM my_index")
    assert res["result"] is None


@pytest.mark.integration
def test_sql_retriever_basic_query(document_store: OpenSearchDocumentStore):
    """Test basic SQL query execution"""
    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
        Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    sql_query = (
        f"SELECT content, category, status, priority FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' ORDER BY priority"
    )
    result = retriever.run(query=sql_query)

    assert "result" in result
    assert len(result["result"]) == 2
    assert isinstance(result["result"], list)
    assert all(isinstance(row, dict) for row in result["result"])

    categories = [row.get("category") for row in result["result"]]
    assert all(cat == "A" for cat in categories)

    for row in result["result"]:
        assert "content" in row
        assert "category" in row
        assert "status" in row
        assert "priority" in row


@pytest.mark.integration
def test_sql_retriever_count_query(document_store: OpenSearchDocumentStore):
    """Test COUNT query execution"""
    docs = [
        Document(content="Doc 1", meta={"category": "A"}),
        Document(content="Doc 2", meta={"category": "B"}),
        Document(content="Doc 3", meta={"category": "A"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    count_query = f"SELECT COUNT(*) as total FROM {document_store._index}"  # noqa: S608
    result = retriever.run(query=count_query)

    assert "result" in result
    assert result["result"] is not None


@pytest.mark.integration
def test_sql_retriever_with_filters(document_store: OpenSearchDocumentStore):
    """Test SQL query with WHERE clause filtering"""

    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    sql_query = (
        f"SELECT content, category, status FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' AND status = 'active'"
    )
    result = retriever.run(query=sql_query)

    assert "result" in result
    assert len(result["result"]) == 1
    assert result["result"][0]["category"] == "A"
    assert result["result"][0]["status"] == "active"


@pytest.mark.integration
def test_sql_retriever_runtime_document_store_switching(
    document_store: OpenSearchDocumentStore, document_store_2: OpenSearchDocumentStore
):
    """Test switching document stores at runtime"""
    docs1 = [
        Document(content="Python programming", meta={"category": "A"}),
        Document(content="Java programming", meta={"category": "B"}),
    ]
    document_store.write_documents(docs1, refresh=True)

    docs2 = [
        Document(content="JavaScript development", meta={"category": "C"}),
        Document(content="TypeScript development", meta={"category": "D"}),
    ]
    document_store_2.write_documents(docs2, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)

    # Query first store
    sql_query1 = f"SELECT content, category FROM {document_store._index} WHERE category = 'A'"  # noqa: S608
    result1 = retriever.run(query=sql_query1)
    assert len(result1["result"]) == 1
    assert "Python" in result1["result"][0]["content"]

    # Query second store at runtime
    sql_query2 = f"SELECT content, category FROM {document_store_2._index} WHERE category = 'C'"  # noqa: S608
    result2 = retriever.run(query=sql_query2, document_store=document_store_2)
    assert len(result2["result"]) == 1
    assert "JavaScript" in result2["result"][0]["content"]

    # Verify results are different
    assert result1["result"][0]["content"] != result2["result"][0]["content"]


@pytest.mark.integration
def test_sql_retriever_error_handling(document_store: OpenSearchDocumentStore):
    """Test error handling for invalid SQL queries"""
    retriever = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=True)

    invalid_query = "SELECT * FROM non_existent_index"
    with pytest.raises(Exception, match="Failed to execute SQL query"):
        retriever.run(query=invalid_query)

    # Test with raise_on_failure=False
    retriever_no_raise = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=False)
    result = retriever_no_raise.run(query=invalid_query)
    assert result["result"] is None


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql_async.return_value = [{"content": "Test doc", "category": "A"}]
    retriever = OpenSearchSQLRetriever(document_store=mock_store)
    res = await retriever.run_async(query="SELECT content, category FROM my_index WHERE category = 'A'")
    mock_store._query_sql_async.assert_called_once_with(
        query="SELECT content, category FROM my_index WHERE category = 'A'",
        cursor=None,
        fetch_size=None,
    )
    assert len(res) == 1
    assert "result" in res
    assert res["result"] == [{"content": "Test doc", "category": "A"}]


@pytest.mark.asyncio
async def test_run_async_with_cursor_and_fetch_size():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql_async.return_value = [{"content": "Test doc", "category": "A"}]
    retriever = OpenSearchSQLRetriever(document_store=mock_store, fetch_size=50)
    res = await retriever.run_async(query="SELECT * FROM my_index", cursor="test_cursor", fetch_size=100)
    mock_store._query_sql_async.assert_called_once_with(
        query="SELECT * FROM my_index",
        cursor="test_cursor",
        fetch_size=100,
    )
    assert res["result"] == [{"content": "Test doc", "category": "A"}]


@pytest.mark.asyncio
async def test_run_async_with_error_raise_on_failure():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql_async.side_effect = Exception("SQL error")
    retriever = OpenSearchSQLRetriever(document_store=mock_store, raise_on_failure=True)
    with pytest.raises(Exception, match="SQL error"):
        await retriever.run_async(query="SELECT * FROM my_index")


@pytest.mark.asyncio
async def test_run_async_with_error_no_raise():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._query_sql_async.side_effect = Exception("SQL error")
    retriever = OpenSearchSQLRetriever(document_store=mock_store, raise_on_failure=False)
    res = await retriever.run_async(query="SELECT * FROM my_index")
    assert res["result"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_basic_query(document_store: OpenSearchDocumentStore):
    """Test basic async SQL query execution"""
    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
    ]
    await document_store.write_documents_async(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    sql_query = (
        f"SELECT content, category, status FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' ORDER BY priority"
    )
    result = await retriever.run_async(query=sql_query)

    assert "result" in result
    assert len(result["result"]) == 2
    assert isinstance(result["result"], list)
    assert all(isinstance(row, dict) for row in result["result"])

    categories = [row.get("category") for row in result["result"]]
    assert all(cat == "A" for cat in categories)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_runtime_document_store_switching(
    document_store: OpenSearchDocumentStore, document_store_2: OpenSearchDocumentStore
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

    retriever = OpenSearchSQLRetriever(document_store=document_store)

    # Query first store
    sql_query1 = f"SELECT content, category FROM {document_store._index} WHERE category = 'A'"  # noqa: S608
    result1 = await retriever.run_async(query=sql_query1)
    assert len(result1["result"]) == 1
    assert "Python" in result1["result"][0]["content"]

    # Query second store at runtime
    sql_query2 = f"SELECT content, category FROM {document_store_2._index} WHERE category = 'C'"  # noqa: S608
    result2 = await retriever.run_async(query=sql_query2, document_store=document_store_2)
    assert len(result2["result"]) == 1
    assert "JavaScript" in result2["result"][0]["content"]

    # Verify results are different
    assert result1["result"][0]["content"] != result2["result"][0]["content"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_error_handling(document_store: OpenSearchDocumentStore):
    """Test async error handling for invalid SQL queries"""
    retriever = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=True)

    invalid_query = "SELECT * FROM non_existent_index"
    with pytest.raises(Exception, match="Failed to execute SQL query"):
        await retriever.run_async(query=invalid_query)

    # Test with raise_on_failure=False
    retriever_no_raise = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=False)
    result = await retriever_no_raise.run_async(query=invalid_query)
    assert result["result"] is None
