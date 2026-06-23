# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.scavio import ScavioWebSearch


class TestScavioWebSearch:
    @pytest.fixture
    def search_response(self):
        return {
            "results": [
                {"title": "Example Title", "url": "https://example.com", "content": "Example content", "position": 1}
            ]
        }

    @pytest.fixture
    def mock_client(self, search_response):
        client = MagicMock()
        client.google.search.return_value = search_response
        return client

    @pytest.fixture
    def mock_async_client(self, search_response):
        client = MagicMock()
        client.google.search = AsyncMock(return_value=search_response)
        return client

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("SCAVIO_API_KEY", "test-key")
        ws = ScavioWebSearch()
        assert ws.top_k == 10
        assert ws.search_params is None
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_params(self):
        ws = ScavioWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            search_params={"country_code": "us"},
        )
        assert ws.top_k == 5
        assert ws.search_params == {"country_code": "us"}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("SCAVIO_API_KEY", "test-key")
        ws = ScavioWebSearch(top_k=5, search_params={"country_code": "us"})
        data = component_to_dict(ws, "ScavioWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.scavio.scavio_websearch.ScavioWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["search_params"] == {"country_code": "us"}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("SCAVIO_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.websearch.scavio.scavio_websearch.ScavioWebSearch",
            "init_parameters": {
                "top_k": 3,
                "search_params": {"language": "en"},
                "api_key": {"env_vars": ["SCAVIO_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        ws = component_from_dict(ScavioWebSearch, data, "ScavioWebSearch")
        assert ws.top_k == 3
        assert ws.search_params == {"language": "en"}

    def test_run_returns_documents_and_links(self, mock_client):
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._scavio_client = mock_client

        result = ws.run(query="test query")

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["links"] == ["https://example.com"]
        mock_client.google.search.assert_called_once_with("test query")

    def test_run_overrides_params_at_runtime(self, mock_client):
        ws = ScavioWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"country_code": "gb"},
        )
        ws._scavio_client = mock_client

        ws.run(query="test", search_params={"country_code": "us"})

        mock_client.google.search.assert_called_once_with("test", country_code="us")

    def test_run_respects_top_k(self, mock_client):
        mock_client.google.search.return_value = {
            "results": [{"title": f"r{i}", "url": f"https://x/{i}", "content": "c"} for i in range(20)]
        }
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"), top_k=3)
        ws._scavio_client = mock_client
        result = ws.run(query="test")
        assert len(result["documents"]) == 3
        assert len(result["links"]) == 3

    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_client):
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._async_scavio_client = mock_async_client

        result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 1
        assert result["links"] == ["https://example.com"]
        mock_async_client.google.search.assert_awaited_once()

    def test_run_raises_on_error(self, mock_client):
        mock_client.google.search.side_effect = Exception("API error")
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
        ws._scavio_client = mock_client

        with pytest.raises(Exception, match="API error"):
            ws.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async_raises_on_error(self, mock_async_client):
        mock_async_client.google.search = AsyncMock(side_effect=Exception("API error"))
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
        ws._async_scavio_client = mock_async_client

        with pytest.raises(Exception, match="API error"):
            await ws.run_async(query="test")

    def test_warm_up_initializes_clients(self):
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
        assert ws._scavio_client is None
        assert ws._async_scavio_client is None
        ws.warm_up()
        assert ws._scavio_client is not None
        assert ws._async_scavio_client is not None

    def test_run_triggers_warm_up(self, search_response):
        with (
            patch("haystack_integrations.components.websearch.scavio.scavio_websearch.ScavioClient") as mock_cls,
            patch("haystack_integrations.components.websearch.scavio.scavio_websearch.AsyncScavioClient"),
        ):
            mock_cls.return_value.google.search.return_value = search_response
            ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
            ws.run(query="test")
            mock_cls.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_run_async_triggers_warm_up(self, search_response):
        with (
            patch("haystack_integrations.components.websearch.scavio.scavio_websearch.ScavioClient"),
            patch("haystack_integrations.components.websearch.scavio.scavio_websearch.AsyncScavioClient") as mock_cls,
        ):
            mock_cls.return_value.google.search = AsyncMock(return_value=search_response)
            ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
            await ws.run_async(query="test")
            mock_cls.assert_called_once_with(api_key="test-key")

    def test_run_empty_results(self, mock_client):
        mock_client.google.search.return_value = {"results": []}
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
        ws._scavio_client = mock_client
        result = ws.run(query="obscure query")
        assert result["documents"] == []
        assert result["links"] == []

    def test_run_raises_runtime_error_when_warm_up_fails_to_initialize_client(self, monkeypatch):
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(ws, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="ScavioWebSearch client failed to initialize"):
            ws.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async_raises_runtime_error_when_warm_up_fails_to_initialize_client(self, monkeypatch):
        ws = ScavioWebSearch(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(ws, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="ScavioWebSearch async client failed to initialize"):
            await ws.run_async(query="test")

    @pytest.mark.skipif(
        not os.environ.get("SCAVIO_API_KEY"),
        reason="Export SCAVIO_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        ws = ScavioWebSearch(api_key=Secret.from_env_var("SCAVIO_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)
