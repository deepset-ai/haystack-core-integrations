# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from haystack_integrations.components.retrievers.supabase import SupabaseGroongaRetriever
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


class TestRetriever:
    def test_init_invalid_store(self):
        with pytest.raises(ValueError, match="document_store must be an instance"):
            SupabaseGroongaRetriever(document_store="not_a_store")

    def test_init(self, groonga_store):
        retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
        assert retriever.top_k == 5
        assert retriever.document_store is groonga_store

    def test_init_default_top_k(self, groonga_store):
        retriever = SupabaseGroongaRetriever(document_store=groonga_store)
        assert retriever.top_k == 10

    def test_run_empty_query(self, groonga_store):
        retriever = SupabaseGroongaRetriever(document_store=groonga_store)
        assert retriever.run(query="") == {"documents": []}

    def test_run(self, groonga_store, mock_supabase_client):
        mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
            data=[{"id": "1", "content": "Python is great", "meta": {}, "score": 1.0}]
        )
        retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
        result = retriever.run(query="Python")
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Python is great"

    @pytest.mark.asyncio
    async def test_run_async(self, groonga_store, mock_supabase_client):
        mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
            data=[{"id": "1", "content": "Python is great", "meta": {}, "score": 1.0}]
        )
        retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
        result = await retriever.run_async(query="Python")
        assert len(result["documents"]) == 1

    @pytest.mark.asyncio
    async def test_run_async_empty_query(self, groonga_store):
        retriever = SupabaseGroongaRetriever(document_store=groonga_store)
        assert await retriever.run_async(query="") == {"documents": []}

    def test_to_dict(self, groonga_store):
        retriever = SupabaseGroongaRetriever(document_store=groonga_store, top_k=5)
        result = retriever.to_dict()
        assert result["init_parameters"]["top_k"] == 5
        assert "document_store" in result["init_parameters"]

    def test_from_dict(self, mock_supabase_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        data = {
            "type": ("haystack_integrations.components.retrievers.supabase.groonga_retriever.SupabaseGroongaRetriever"),
            "init_parameters": {
                "top_k": 7,
                "filters": {},
                "filter_policy": "replace",
                "document_store": {
                    "type": (
                        "haystack_integrations.document_stores.supabase"
                        ".groonga_document_store.SupabaseGroongaDocumentStore"
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
