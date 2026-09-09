# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.tavily import TavilyWebSearch


class TestInitializationAndSerialization:
    def test_init_default(self):
        ws = TavilyWebSearch()
        assert ws.top_k == 10
        assert ws.search_params is None
        assert ws.api_key == Secret.from_env_var("TAVILY_API_KEY")
        assert ws._tavily_client is None
        assert ws._async_tavily_client is None

    def test_init_with_params(self):
        ws = TavilyWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            search_params={"search_depth": "advanced"},
        )
        assert ws.top_k == 5
        assert ws.search_params == {"search_depth": "advanced"}

    def test_to_dict(self):
        ws = TavilyWebSearch(top_k=5, search_params={"search_depth": "advanced"})
        data = component_to_dict(ws, "TavilyWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.tavily.tavily_websearch.TavilyWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["search_params"] == {"search_depth": "advanced"}

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.websearch.tavily.tavily_websearch.TavilyWebSearch",
            "init_parameters": {
                "top_k": 3,
                "search_params": {"include_answer": True},
                "api_key": {"env_vars": ["TAVILY_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        ws = component_from_dict(TavilyWebSearch, data, "TavilyWebSearch")
        assert ws.top_k == 3
        assert ws.search_params == {"include_answer": True}


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        component = TavilyWebSearch()

        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            component.warm_up()

    @patch("haystack_integrations.components.websearch.tavily.tavily_websearch.TavilyClient")
    def test_sync_lifecycle(self, mock_client_cls):
        component = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        client = mock_client_cls.return_value

        component.warm_up()
        assert component._tavily_client is client
        assert component._async_tavily_client is None

        component.close()
        client.close.assert_called_once_with()
        assert component._tavily_client is None

        component.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.websearch.tavily.tavily_websearch.AsyncTavilyClient")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, mock_client_cls):
        component = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        client = mock_client_cls.return_value
        client.close = AsyncMock()

        await component.warm_up_async()
        assert component._async_tavily_client is client
        assert component._tavily_client is None

        await component.close_async()
        client.close.assert_awaited_once_with()
        assert component._async_tavily_client is None

        await component.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.websearch.tavily.tavily_websearch.TavilyClient")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        component = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        component.warm_up()
        component.warm_up()
        mock_client_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @patch("haystack_integrations.components.websearch.tavily.tavily_websearch.AsyncTavilyClient")
    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        component = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        await component.warm_up_async()
        await component.warm_up_async()
        mock_client_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        component = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        component.close()
        await component.close_async()
        assert component._tavily_client is None
        assert component._async_tavily_client is None

    @patch("haystack_integrations.components.websearch.tavily.tavily_websearch.AsyncTavilyClient")
    @patch("haystack_integrations.components.websearch.tavily.tavily_websearch.TavilyClient")
    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        component = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        sync_client = mock_sync_cls.return_value
        async_client = mock_async_cls.return_value
        async_client.close = AsyncMock()
        component.warm_up()
        await component.warm_up_async()

        component.close()
        assert component._tavily_client is None
        assert component._async_tavily_client is async_client
        async_client.close.assert_not_awaited()

        await component.close_async()
        assert component._async_tavily_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    @pytest.fixture
    def search_response(self):
        return {
            "results": [
                {"title": "Example Title", "url": "https://example.com", "content": "Example content", "score": 0.95}
            ]
        }

    @pytest.fixture
    def mock_client(self, search_response):
        client = MagicMock()
        client.search.return_value = search_response
        return client

    @pytest.fixture
    def mock_async_client(self, search_response):
        client = MagicMock()
        client.search = AsyncMock(return_value=search_response)
        return client

    def test_run_returns_documents_and_links(self, mock_client):
        ws = TavilyWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._tavily_client = mock_client

        result = ws.run(query="test query")

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["documents"][0].score == 0.95
        assert result["links"] == ["https://example.com"]
        mock_client.search.assert_called_once_with(query="test query", max_results=10)

    def test_run_overrides_params_at_runtime(self, mock_client):
        ws = TavilyWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"search_depth": "basic"},
        )
        ws._tavily_client = mock_client

        ws.run(query="test", search_params={"search_depth": "advanced", "max_results": 3})

        mock_client.search.assert_called_once_with(query="test", search_depth="advanced", max_results=3)

    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_client):
        ws = TavilyWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._async_tavily_client = mock_async_client

        result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 1
        assert result["links"] == ["https://example.com"]
        mock_async_client.search.assert_awaited_once()

    def test_run_raises_on_error(self, mock_client):
        mock_client.search.side_effect = Exception("API error")
        ws = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        ws._tavily_client = mock_client

        with pytest.raises(Exception, match="API error"):
            ws.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async_raises_on_error(self, mock_async_client):
        mock_async_client.search = AsyncMock(side_effect=Exception("API error"))
        ws = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        ws._async_tavily_client = mock_async_client

        with pytest.raises(Exception, match="API error"):
            await ws.run_async(query="test")

    def test_run_triggers_warm_up(self, search_response):
        with patch("haystack_integrations.components.websearch.tavily.tavily_websearch.TavilyClient") as mock_cls:
            mock_cls.return_value.search.return_value = search_response
            ws = TavilyWebSearch(api_key=Secret.from_token("test-key"))
            ws.run(query="test")
            mock_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @pytest.mark.asyncio
    async def test_run_async_triggers_warm_up(self, search_response):
        with patch("haystack_integrations.components.websearch.tavily.tavily_websearch.AsyncTavilyClient") as mock_cls:
            mock_cls.return_value.search = AsyncMock(return_value=search_response)
            ws = TavilyWebSearch(api_key=Secret.from_token("test-key"))
            await ws.run_async(query="test")
            mock_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    def test_run_empty_results(self, mock_client):
        mock_client.search.return_value = {"results": []}
        ws = TavilyWebSearch(api_key=Secret.from_token("test-key"))
        ws._tavily_client = mock_client
        result = ws.run(query="obscure query")
        assert result["documents"] == []
        assert result["links"] == []


@pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"),
    reason="Export TAVILY_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestIntegration:
    def test_run_integration(self):
        ws = TavilyWebSearch(api_key=Secret.from_env_var("TAVILY_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)

    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ws = TavilyWebSearch(api_key=Secret.from_env_var("TAVILY_API_KEY"), top_k=3)
        result = await ws.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
