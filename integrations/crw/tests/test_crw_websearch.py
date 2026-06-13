# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.crw import CRWWebSearch


class TestCRWWebSearch:
    @pytest.fixture
    def search_response(self):
        return [
            {"title": "Example Title", "url": "https://example.com", "description": "Example content"}
        ]

    @pytest.fixture
    def mock_client(self, search_response):
        client = MagicMock()
        client.search.return_value = search_response
        return client

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("CRW_API_KEY", "test-key")
        ws = CRWWebSearch()
        assert ws.top_k == 10
        assert ws.api_url is None
        assert ws.search_params is None
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_params(self):
        ws = CRWWebSearch(
            api_key=Secret.from_token("custom-key"),
            api_url="http://localhost:3000",
            top_k=5,
            search_params={"sources": ["web"]},
        )
        assert ws.top_k == 5
        assert ws.api_url == "http://localhost:3000"
        assert ws.search_params == {"sources": ["web"]}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("CRW_API_KEY", "test-key")
        ws = CRWWebSearch(top_k=5, search_params={"sources": ["web"]})
        data = component_to_dict(ws, "CRWWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.crw.crw_websearch.CRWWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["search_params"] == {"sources": ["web"]}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("CRW_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.websearch.crw.crw_websearch.CRWWebSearch",
            "init_parameters": {
                "top_k": 3,
                "search_params": {"sources": ["web"]},
                "api_key": {"env_vars": ["CRW_API_KEY"], "strict": False, "type": "env_var"},
            },
        }
        ws = component_from_dict(CRWWebSearch, data, "CRWWebSearch")
        assert ws.top_k == 3
        assert ws.search_params == {"sources": ["web"]}

    def test_run_returns_documents_and_links(self, mock_client):
        ws = CRWWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._crw_client = mock_client

        result = ws.run(query="test query")

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["links"] == ["https://example.com"]
        mock_client.search.assert_called_once_with("test query", limit=10)

    def test_run_prefers_markdown_content(self, mock_client):
        mock_client.search.return_value = [
            {
                "title": "Example Title",
                "url": "https://example.com",
                "description": "short description",
                "markdown": "# Full markdown content",
            }
        ]
        ws = CRWWebSearch(api_key=Secret.from_token("test-key"))
        ws._crw_client = mock_client

        result = ws.run(query="test query")

        assert result["documents"][0].content == "# Full markdown content"

    def test_run_overrides_params_at_runtime(self, mock_client):
        ws = CRWWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"sources": ["web"]},
        )
        ws._crw_client = mock_client

        ws.run(query="test", search_params={"sources": ["web", "images"], "limit": 3})

        mock_client.search.assert_called_once_with("test", sources=["web", "images"], limit=3)

    def test_run_raises_on_error(self, mock_client):
        mock_client.search.side_effect = Exception("API error")
        ws = CRWWebSearch(api_key=Secret.from_token("test-key"))
        ws._crw_client = mock_client

        with pytest.raises(Exception, match="API error"):
            ws.run(query="test")

    def test_warm_up_initializes_client(self):
        with patch("haystack_integrations.components.websearch.crw.crw_websearch.CrwClient") as mock_cls:
            ws = CRWWebSearch(api_key=Secret.from_token("test-key"), api_url="http://localhost:3000")
            assert ws._crw_client is None
            ws.warm_up()
            assert ws._crw_client is not None
            mock_cls.assert_called_once_with(api_url="http://localhost:3000", api_key="test-key")

    def test_run_triggers_warm_up(self, search_response):
        with patch("haystack_integrations.components.websearch.crw.crw_websearch.CrwClient") as mock_cls:
            mock_cls.return_value.search.return_value = search_response
            ws = CRWWebSearch(api_key=Secret.from_token("test-key"))
            ws.run(query="test")
            mock_cls.assert_called_once_with(api_url=None, api_key="test-key")

    def test_run_empty_results(self, mock_client):
        mock_client.search.return_value = []
        ws = CRWWebSearch(api_key=Secret.from_token("test-key"))
        ws._crw_client = mock_client
        result = ws.run(query="obscure query")
        assert result["documents"] == []
        assert result["links"] == []

    def test_run_raises_runtime_error_when_warm_up_fails_to_initialize_client(self, monkeypatch):
        ws = CRWWebSearch(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(ws, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="CRWWebSearch client failed to initialize"):
            ws.run(query="test")

    @pytest.mark.skipif(
        not os.environ.get("CRW_API_KEY"),
        reason="Export CRW_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        ws = CRWWebSearch(api_key=Secret.from_env_var("CRW_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)
