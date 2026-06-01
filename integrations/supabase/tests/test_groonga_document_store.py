# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore


@pytest.fixture
def mock_supabase_client():
    """Creates a mock Supabase client so we never hit a real database."""
    with patch("haystack_integrations.document_stores.supabase.groonga_document_store.create_client") as mock_create:
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        mock_client.rpc.return_value.execute.return_value = MagicMock(data=[], count=0)

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
    store.warm_up()
    return store

class TestDocumentStore:
    def test_init_defaults(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store.table_name == "haystack_groonga_documents"
        assert store.recreate_table is False
        assert store.supabase_url == "https://fake-project.supabase.co"
        assert store._client is None

    def test_init_custom_params(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake-project.supabase.co",
            table_name="my_custom_table",
            recreate_table=True,
        )
        assert store.table_name == "my_custom_table"
        assert store.recreate_table is True
        assert store._client is None

    def test_invalid_table_name_raises(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        with pytest.raises(ValueError, match="Invalid table_name"):
            SupabaseGroongaDocumentStore(
                supabase_url="https://fake-project.supabase.co",
                table_name="bad-name; DROP TABLE users;",
            )

    def test_table_name_with_numbers_allowed(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake-project.supabase.co",
            table_name="my_table_123",
        )
        assert store.table_name == "my_table_123"

    def test_warm_up_initializes_client(self, mock_supabase_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store._client is None
        store.warm_up()
        assert store._client is not None

    def test_count_documents(self, groonga_store, mock_supabase_client):
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = MagicMock(count=5)
        assert groonga_store.count_documents() == 5

    def test_count_documents_empty(self, groonga_store, mock_supabase_client):
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = MagicMock(count=0)
        assert groonga_store.count_documents() == 0

    def test_write_documents(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        mock_table.insert.return_value.execute.return_value = MagicMock(data=[{}])

        documents = [
            Document(content="Python is great"),
            Document(content="Haystack is a RAG framework"),
        ]
        written = groonga_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
        assert written == 2

    def test_write_documents_empty(self, groonga_store):
        assert groonga_store.write_documents([]) == 0

    def test_write_documents_overwrite(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.upsert.return_value.execute.return_value = MagicMock(data=[{}])

        written = groonga_store.write_documents([Document(content="test document")], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        mock_table.upsert.assert_called_once()

    def test_write_documents_skip_existing(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "existing"}])

        written = groonga_store.write_documents([Document(content="already exists")], policy=DuplicatePolicy.SKIP)
        assert written == 0

    def test_write_documents_fail_on_duplicate(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "existing"}])

        with pytest.raises(DuplicateDocumentError):
            groonga_store.write_documents([Document(content="duplicate doc")], policy=DuplicatePolicy.FAIL)

    def test_delete_all_documents(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.delete.return_value.neq.return_value.execute.return_value = MagicMock(data=[])

        groonga_store.delete_all_documents()
        mock_table.delete.assert_called_once()

    def test_delete_documents(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.delete.return_value.in_.return_value.execute.return_value = MagicMock(data=[])

        groonga_store.delete_documents(["id1", "id2"])
        mock_table.delete.assert_called_once()

    def test_delete_documents_empty(self, groonga_store, mock_supabase_client):
        groonga_store.delete_documents([])
        mock_supabase_client.table.return_value.delete.assert_not_called()

    def test_filter_documents(self, groonga_store, mock_supabase_client):
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

    def test_filter_documents_with_filters(self, groonga_store, mock_supabase_client):
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": "1", "content": "Python is great", "meta": {"language": "en"}, "score": None}]
        )
        filters = {"operator": "AND", "conditions": [{"field": "meta.language", "operator": "==", "value": "en"}]}
        docs = groonga_store.filter_documents(filters=filters)
        assert len(docs) == 1

    def test_to_dict(self, groonga_store):
        result = groonga_store.to_dict()
        assert result["type"] == (
            "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
        )
        assert result["init_parameters"]["table_name"] == "test_groonga_documents"
        assert result["init_parameters"]["supabase_url"] == "https://fake-project.supabase.co"
        assert result["init_parameters"]["recreate_table"] is False

    def test_from_dict(self, mock_supabase_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        data = {
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
        }
        store = SupabaseGroongaDocumentStore.from_dict(data)
        assert store.table_name == "test_groonga_documents"
        assert store.supabase_url == "https://fake-project.supabase.co"
        assert store._client is None
