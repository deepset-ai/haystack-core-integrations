# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.supabase import SupabaseGroongaBM25Retriever


class TestInit:
    def test_init_invalid_store(self):
        with pytest.raises(ValueError, match="document_store must be an instance"):
            SupabaseGroongaBM25Retriever(document_store="not_a_store")

    def test_init_invalid_top_k(self, mock_groonga_store):
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, top_k=0)

    def test_init(self, mock_groonga_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, top_k=5)
        assert retriever.top_k == 5
        assert retriever.document_store is mock_groonga_store

    def test_init_default_top_k(self, mock_groonga_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store)
        assert retriever.top_k == 10


class TestRun:
    def test_run_empty_query(self, mock_groonga_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store)
        assert retriever.run(query="") == {"documents": []}

    def test_run(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_client.rpc.return_value.execute.return_value = MagicMock(
            data=[{"id": "1", "content": "Python is great", "meta": {}, "score": 1.0}]
        )
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, top_k=5)
        result = retriever.run(query="Python")
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Python is great"

    async def test_run_async_uses_async_retrieval(self, mock_groonga_store):
        """run_async() must delegate to _groonga_retrieval_async, not the sync path."""
        expected_doc = Document(id="1", content="Python is great")
        mock_groonga_store._groonga_retrieval_async = AsyncMock(return_value=[expected_doc])

        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, top_k=5)
        result = await retriever.run_async(query="Python")

        mock_groonga_store._groonga_retrieval_async.assert_awaited_once()
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Python is great"

    async def test_run_async_empty_query(self, mock_groonga_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store)
        assert await retriever.run_async(query="") == {"documents": []}

    def test_run_ignore_errors(self, mock_groonga_store):
        mock_groonga_store._groonga_retrieval = MagicMock(side_effect=Exception("DB error"))
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, raise_on_failure=False)
        assert retriever.run(query="test") == {"documents": []}

    def test_run_raises_on_failure_by_default(self, mock_groonga_store):
        mock_groonga_store._groonga_retrieval = MagicMock(side_effect=Exception("DB error"))
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store)
        with pytest.raises(Exception, match="DB error"):
            retriever.run(query="test")

    async def test_run_async_ignore_errors(self, mock_groonga_store):
        mock_groonga_store._groonga_retrieval_async = AsyncMock(side_effect=Exception("async DB error"))
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, raise_on_failure=False)
        assert await retriever.run_async(query="test") == {"documents": []}

    async def test_run_async_raises_on_failure_by_default(self, mock_groonga_store):
        mock_groonga_store._groonga_retrieval_async = AsyncMock(side_effect=Exception("async DB error"))
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store)
        with pytest.raises(Exception, match="async DB error"):
            await retriever.run_async(query="test")


class TestSerde:
    def test_to_dict(self, mock_groonga_store):
        retriever = SupabaseGroongaBM25Retriever(document_store=mock_groonga_store, top_k=5)
        result = retriever.to_dict()
        assert result["init_parameters"]["top_k"] == 5
        assert "document_store" in result["init_parameters"]

    def test_from_dict(self, mock_groonga_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        data = {
            "type": (
                "haystack_integrations.components.retrievers.supabase.groonga_bm25_retriever.SupabaseGroongaBM25Retriever"
            ),
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
        retriever = SupabaseGroongaBM25Retriever.from_dict(data)
        assert retriever.top_k == 7
