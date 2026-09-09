# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.fetchers.tavily import TavilyFetcher


class TestInitializationAndSerialization:
    def test_init_default(self):
        fetcher = TavilyFetcher()
        assert fetcher.extract_depth == "basic"
        assert fetcher.include_images is False
        assert fetcher.extract_params is None
        assert fetcher.api_key == Secret.from_env_var("TAVILY_API_KEY")
        assert fetcher._tavily_client is None
        assert fetcher._async_tavily_client is None

    def test_init_with_params(self):
        fetcher = TavilyFetcher(
            api_key=Secret.from_token("custom-key"),
            extract_depth="advanced",
            include_images=True,
            extract_params={"format": "text"},
        )
        assert fetcher.extract_depth == "advanced"
        assert fetcher.include_images is True
        assert fetcher.extract_params == {"format": "text"}

    def test_to_dict(self):
        fetcher = TavilyFetcher(extract_depth="advanced", extract_params={"format": "text"})
        data = component_to_dict(fetcher, "TavilyFetcher")
        assert data["type"] == "haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyFetcher"
        assert data["init_parameters"]["extract_depth"] == "advanced"
        assert data["init_parameters"]["extract_params"] == {"format": "text"}

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyFetcher",
            "init_parameters": {
                "extract_depth": "advanced",
                "include_images": True,
                "extract_params": {"format": "text"},
                "api_key": {"env_vars": ["TAVILY_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        fetcher = component_from_dict(TavilyFetcher, data, "TavilyFetcher")
        assert fetcher.extract_depth == "advanced"
        assert fetcher.include_images is True
        assert fetcher.extract_params == {"format": "text"}


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        component = TavilyFetcher()

        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            component.warm_up()

    @patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyClient")
    def test_sync_lifecycle(self, mock_client_cls):
        component = TavilyFetcher(api_key=Secret.from_token("test-key"))
        client = mock_client_cls.return_value

        component.warm_up()
        assert component._tavily_client is client
        assert component._async_tavily_client is None

        component.close()
        client.close.assert_called_once_with()
        assert component._tavily_client is None

        component.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.AsyncTavilyClient")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, mock_client_cls):
        component = TavilyFetcher(api_key=Secret.from_token("test-key"))
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

    @patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyClient")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        component = TavilyFetcher(api_key=Secret.from_token("test-key"))
        component.warm_up()
        component.warm_up()
        mock_client_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.AsyncTavilyClient")
    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        component = TavilyFetcher(api_key=Secret.from_token("test-key"))
        await component.warm_up_async()
        await component.warm_up_async()
        mock_client_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        component = TavilyFetcher(api_key=Secret.from_token("test-key"))
        component.close()
        await component.close_async()
        assert component._tavily_client is None
        assert component._async_tavily_client is None

    @patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.AsyncTavilyClient")
    @patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyClient")
    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        component = TavilyFetcher(api_key=Secret.from_token("test-key"))
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
    def extract_response(self):
        return {
            "results": [
                {
                    "url": "https://example.com",
                    "raw_content": "Example page content",
                    "images": ["https://example.com/image.png"],
                }
            ],
            "failed_results": [],
            "response_time": 1.23,
            "usage": {"credits": 1},
            "request_id": "req-abc123",
        }

    @pytest.fixture
    def mock_client(self, extract_response):
        client = MagicMock()
        client.extract.return_value = extract_response
        return client

    @pytest.fixture
    def mock_async_client(self, extract_response):
        client = MagicMock()
        client.extract = AsyncMock(return_value=extract_response)
        return client

    def test_run_returns_documents_and_meta(self, mock_client):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        fetcher._tavily_client = mock_client

        result = fetcher.run(urls=["https://example.com"])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert isinstance(doc, Document)
        assert doc.content == "Example page content"
        assert doc.meta["url"] == "https://example.com"
        assert "images" not in doc.meta  # include_images=False by default

        meta = result["meta"]
        assert meta["response_time"] == 1.23
        assert meta["usage"] == {"credits": 1}
        assert meta["request_id"] == "req-abc123"
        assert meta["failed_results"] == []

    def test_run_includes_images_when_enabled(self, mock_client):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"), include_images=True)
        fetcher._tavily_client = mock_client

        result = fetcher.run(urls=["https://example.com"])

        assert result["documents"][0].meta["images"] == ["https://example.com/image.png"]

    def test_run_passes_extract_depth(self, mock_client):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"), extract_depth="advanced")
        fetcher._tavily_client = mock_client

        fetcher.run(urls=["https://example.com"])

        mock_client.extract.assert_called_once_with(
            urls=["https://example.com"],
            extract_depth="advanced",
            include_images=False,
        )

    def test_run_passes_init_extract_params(self, mock_client):
        fetcher = TavilyFetcher(
            api_key=Secret.from_token("test-key"),
            extract_params={"format": "text"},
        )
        fetcher._tavily_client = mock_client

        fetcher.run(urls=["https://example.com"])

        mock_client.extract.assert_called_once_with(
            urls=["https://example.com"],
            extract_depth="basic",
            include_images=False,
            format="text",
        )

    def test_run_overrides_extract_params_at_runtime(self, mock_client):
        fetcher = TavilyFetcher(
            api_key=Secret.from_token("test-key"),
            extract_params={"format": "text"},
        )
        fetcher._tavily_client = mock_client

        fetcher.run(urls=["https://example.com"], extract_params={"format": "markdown", "include_favicon": True})

        mock_client.extract.assert_called_once_with(
            urls=["https://example.com"],
            extract_depth="basic",
            include_images=False,
            format="markdown",
            include_favicon=True,
        )

    def test_run_multiple_urls(self, mock_client):
        mock_client.extract.return_value = {
            "results": [
                {"url": "https://a.com", "raw_content": "Content A"},
                {"url": "https://b.com", "raw_content": "Content B"},
            ],
            "failed_results": [],
            "response_time": 2.0,
            "usage": None,
            "request_id": None,
        }
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        fetcher._tavily_client = mock_client

        result = fetcher.run(urls=["https://a.com", "https://b.com"])

        assert len(result["documents"]) == 2
        assert result["documents"][0].meta["url"] == "https://a.com"
        assert result["documents"][1].meta["url"] == "https://b.com"

    def test_run_with_failed_results_in_meta(self, mock_client):
        mock_client.extract.return_value = {
            "results": [],
            "failed_results": [{"url": "https://broken.com", "error": "Connection refused"}],
            "response_time": 0.5,
            "usage": None,
            "request_id": "req-xyz",
        }
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        fetcher._tavily_client = mock_client

        result = fetcher.run(urls=["https://broken.com"])

        assert result["documents"] == []
        assert result["meta"]["failed_results"] == [{"url": "https://broken.com", "error": "Connection refused"}]

    def test_run_empty_results(self, mock_client):
        mock_client.extract.return_value = {
            "results": [],
            "failed_results": [],
            "response_time": 0.1,
            "usage": None,
            "request_id": None,
        }
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        fetcher._tavily_client = mock_client

        result = fetcher.run(urls=["https://example.com"])

        assert result["documents"] == []

    def test_run_triggers_warm_up(self, extract_response):
        with patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyClient") as mock_cls:
            mock_cls.return_value.extract.return_value = extract_response
            fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
            fetcher.run(urls=["https://example.com"])
            mock_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_client):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        fetcher._async_tavily_client = mock_async_client

        result = await fetcher.run_async(urls=["https://example.com"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Example page content"
        mock_async_client.extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_async_triggers_warm_up(self, extract_response):
        with patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.AsyncTavilyClient") as mock_cls:
            mock_cls.return_value.extract = AsyncMock(return_value=extract_response)
            fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
            await fetcher.run_async(urls=["https://example.com"])
            mock_cls.assert_called_once_with(api_key="test-key", client_name="haystack")

    @pytest.mark.asyncio
    async def test_run_async_raises_on_error(self, mock_async_client):
        mock_async_client.extract = AsyncMock(side_effect=Exception("API error"))
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        fetcher._async_tavily_client = mock_async_client

        with pytest.raises(Exception, match="API error"):
            await fetcher.run_async(urls=["https://example.com"])


@pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"),
    reason="Export TAVILY_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestIntegration:
    def test_run_integration(self):
        fetcher = TavilyFetcher(api_key=Secret.from_env_var("TAVILY_API_KEY"))
        result = fetcher.run(urls=["https://haystack.deepset.ai"])
        assert len(result["documents"]) > 0
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content
        assert result["meta"]["response_time"] is not None

    def test_run_integration_pdf(self):
        # Attention Is All You Need — stable public arXiv PDF
        fetcher = TavilyFetcher(api_key=Secret.from_env_var("TAVILY_API_KEY"))
        result = fetcher.run(urls=["https://arxiv.org/pdf/1706.03762"])
        assert len(result["documents"]) > 0
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content
        assert result["meta"]["response_time"] is not None

    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        fetcher = TavilyFetcher(api_key=Secret.from_env_var("TAVILY_API_KEY"))
        result = await fetcher.run_async(urls=["https://haystack.deepset.ai"])
        assert len(result["documents"]) > 0
        assert result["meta"]["response_time"] is not None
