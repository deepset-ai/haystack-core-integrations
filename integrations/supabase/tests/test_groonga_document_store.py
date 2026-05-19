# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.supabase import SupabaseGroongaRetriever
from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────


@pytest.fixture
def mock_supabase_client():
    """Creates a mock Supabase client so we never hit a real database."""
    with patch("haystack_integrations.document_stores.supabase.groonga_document_store.create_client") as mock_create:
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        # Mock rpc calls (used in _setup_table)
        mock_client.rpc.return_value.execute.return_value = MagicMock(data=[], count=0)

        # Mock table calls
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.delete.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.in_.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[], count=0)

        yield mock_client


@pytest.fixture
def groonga_store(mock_supabase_client, monkeypatch):  # noqa: ARG001
    """Creates a SupabaseGroongaDocumentStore with mocked client and calls warm_up()."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    store = SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
        table_name="test_groonga_documents",
        recreate_table=False,
    )
    store.warm_up()  # must call warm_up() before using the store
    return store


# ─────────────────────────────────────────────
# DOCUMENT STORE TESTS
# ─────────────────────────────────────────────


def test_init_defaults(monkeypatch):
    """Test that default parameters are set correctly without connecting."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    store = SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
    )
    assert store.table_name == "haystack_groonga_documents"
    assert store.recreate_table is False
    assert store.supabase_url == "https://fake-project.supabase.co"
    # client should be None before warm_up()
    assert store._client is None


def test_init_custom_params(monkeypatch):
    """Test that custom parameters are set correctly without connecting."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    store = SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
        table_name="my_custom_table",
        recreate_table=True,
    )
    assert store.table_name == "my_custom_table"
    assert store.recreate_table is True
    # client should be None before warm_up()
    assert store._client is None


def test_warm_up_initializes_client(mock_supabase_client, monkeypatch):  # noqa: ARG001
    """Test that warm_up() initializes the client."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    store = SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
    )
    assert store._client is None
    store.warm_up()
    assert store._client is not None


def test_init_invalid_store():
    """Test that passing wrong store to retriever raises ValueError."""
    with pytest.raises(ValueError, match="document_store must be an instance"):
        SupabaseGroongaRetriever(document_store="not_a_store")


def test_count_documents(groonga_store, mock_supabase_client):
    """Test count_documents returns correct number."""
    mock_supabase_client.table.return_value.select.return_value.execute.return_value = MagicMock(count=5)
    count = groonga_store.count_documents()
    assert count == 5


def test_count_documents_empty(groonga_store, mock_supabase_client):
    """Test count_documents returns 0 when store is empty."""
    mock_supabase_client.table.return_value.select.return_value.execute.return_value = MagicMock(count=0)
    count = groonga_store.count_documents()
    assert count == 0


def test_write_documents(groonga_store, mock_supabase_client):
    """Test that write_documents writes correct number of documents."""
    mock_table = mock_supabase_client.table.return_value
    # mock select chain so no existing docs are found
    mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    mock_table.insert.return_value.execute.return_value = MagicMock(data=[{}])

    documents = [
        Document(content="Python is great"),
        Document(content="Haystack is a RAG framework"),
    ]
    # use OVERWRITE to avoid the duplicate check entirely
    written = groonga_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    assert written == 2


def test_write_documents_empty(groonga_store):
    """Test that writing empty list returns 0."""
    written = groonga_store.write_documents([])
    assert written == 0


def test_write_documents_overwrite(groonga_store, mock_supabase_client):
    """Test that overwrite policy uses upsert."""
    mock_table = mock_supabase_client.table.return_value
    mock_table.upsert.return_value.execute.return_value = MagicMock(data=[{}])

    documents = [Document(content="test document")]
    written = groonga_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    assert written == 1
    mock_table.upsert.assert_called_once()


def test_write_documents_skip(groonga_store, mock_supabase_client):
    """Test that skip policy skips existing documents."""
    mock_table = mock_supabase_client.table.return_value
    # simulate document already exists
    mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "existing"}])

    documents = [Document(content="already exists")]
    written = groonga_store.write_documents(documents, policy=DuplicatePolicy.SKIP)
    assert written == 0


def test_write_documents_fail_on_duplicate(groonga_store, mock_supabase_client):
    """Test that FAIL policy raises error on duplicate."""
    mock_table = mock_supabase_client.table.return_value
    # simulate document already exists
    mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "existing"}])

    doc = Document(content="duplicate doc")
    with pytest.raises(DuplicateDocumentError):
        groonga_store.write_documents([doc], policy=DuplicatePolicy.FAIL)


def test_delete_documents(groonga_store, mock_supabase_client):
    """Test that delete_documents calls delete with correct IDs."""
    mock_table = mock_supabase_client.table.return_value
    mock_table.delete.return_value.in_.return_value.execute.return_value = MagicMock(data=[])

    groonga_store.delete_documents(["id1", "id2"])
    mock_table.delete.assert_called_once()


def test_delete_documents_empty(groonga_store, mock_supabase_client):
    """Test that deleting empty list does nothing."""
    groonga_store.delete_documents([])
    mock_supabase_client.table.return_value.delete.assert_not_called()


def test_filter_documents(groonga_store, mock_supabase_client):
    """Test that filter_documents returns correct documents."""
    mock_supabase_client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=[
            {"id": "1", "content": "Python is great", "meta": {}, "score": None},
            {"id": "2", "content": "Haystack rocks", "meta": {}, "score": None},
        ]
    )
    docs = groonga_store.filter_documents()
    assert len(docs) == 2
    assert docs[0].content == "Python is great"
    assert docs[1].content == "Haystack rocks"


def test_filter_documents_with_filters(groonga_store, mock_supabase_client):
    """Test that filter_documents applies filters correctly."""
    mock_table = mock_supabase_client.table.return_value
    mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[
            {"id": "1", "content": "Python is great", "meta": {"language": "en"}, "score": None},
        ]
    )
    filters = {"conditions": [{"field": "meta.language", "operator": "==", "value": "en"}]}
    docs = groonga_store.filter_documents(filters=filters)
    assert len(docs) == 1


# ─────────────────────────────────────────────
# SERIALIZATION TESTS
# ─────────────────────────────────────────────


def test_to_dict(groonga_store):
    """Test that to_dict returns correct dictionary."""
    result = groonga_store.to_dict()
    assert result["type"] == (
        "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
    )
    assert result["init_parameters"]["table_name"] == "test_groonga_documents"
    assert result["init_parameters"]["supabase_url"] == "https://fake-project.supabase.co"
    assert result["init_parameters"]["recreate_table"] is False


def test_from_dict(mock_supabase_client, monkeypatch):  # noqa: ARG001
    """Test that from_dict recreates the store correctly."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    data = {
        "type": ("haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"),
        "init_parameters": {
            "supabase_url": "https://fake-project.supabase.co",
            "supabase_key": {
                "type": "env_var",
                "env_vars": ["SUPABASE_SERVICE_KEY"],
                "strict": True,
            },
            "table_name": "test_groonga_documents",
            "recreate_table": False,
        },
    }
    store = SupabaseGroongaDocumentStore.from_dict(data)
    assert store.table_name == "test_groonga_documents"
    assert store.supabase_url == "https://fake-project.supabase.co"
    # client should be None — warm_up() not called yet
    assert store._client is None


# ─────────────────────────────────────────────
# RETRIEVER TESTS
# ─────────────────────────────────────────────


def test_retriever_init(groonga_store):
    """Test that retriever initializes correctly."""
    retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
    assert retriever.top_k == 5
    assert retriever.document_store == groonga_store


def test_retriever_init_default_top_k(groonga_store):
    """Test that retriever default top_k is 10."""
    retriever = SupabaseGroongaRetriever(document_store=groonga_store)
    assert retriever.top_k == 10


def test_retriever_run_empty_query(groonga_store):
    """Test that empty query returns empty documents."""
    retriever = SupabaseGroongaRetriever(document_store=groonga_store)
    result = retriever.run(query="")
    assert result == {"documents": []}


def test_retriever_run(groonga_store, mock_supabase_client):
    """Test that retriever run calls document store correctly."""
    mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
        data=[
            {"id": "1", "content": "Python is great", "meta": {}, "score": 1.0},
        ]
    )
    retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
    result = retriever.run(query="Python")
    assert "documents" in result
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "Python is great"


@pytest.mark.asyncio
async def test_retriever_run_async(groonga_store, mock_supabase_client):
    """Test that async retriever run returns same result as sync run."""
    mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
        data=[
            {"id": "1", "content": "Python is great", "meta": {}, "score": 1.0},
        ]
    )
    retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
    result = await retriever.run_async(query="Python")
    assert "documents" in result
    assert len(result["documents"]) == 1


@pytest.mark.asyncio
async def test_retriever_run_async_empty_query(groonga_store):
    """Test that empty query in async run returns empty documents."""
    retriever = SupabaseGroongaRetriever(document_store=groonga_store)
    result = await retriever.run_async(query="")
    assert result == {"documents": []}


def test_retriever_to_dict(groonga_store):
    """Test that retriever serializes correctly."""
    retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
    result = retriever.to_dict()
    assert result["init_parameters"]["top_k"] == 5
    assert "document_store" in result["init_parameters"]


def test_retriever_from_dict(mock_supabase_client, monkeypatch):  # noqa: ARG001
    """Test that retriever deserializes correctly."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    data = {
        "type": ("haystack_integrations.components.retrievers.supabase.groonga_retriever.SupabaseGroongaRetriever"),
        "init_parameters": {
            "top_k": 7,
            "filters": {},
            "filter_policy": "replace",
            "document_store": {
                "type": (
                    "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
                ),
                "init_parameters": {
                    "supabase_url": "https://fake-project.supabase.co",
                    "supabase_key": {
                        "type": "env_var",
                        "env_vars": ["SUPABASE_SERVICE_KEY"],
                        "strict": True,
                    },
                    "table_name": "test_groonga_documents",
                    "recreate_table": False,
                },
            },
        },
    }
    retriever = SupabaseGroongaRetriever.from_dict(data)
    assert retriever.top_k == 7
