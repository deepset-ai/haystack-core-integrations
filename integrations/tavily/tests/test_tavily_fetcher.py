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


class TestTavilyFetcher:
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

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        fetcher = TavilyFetcher()
        assert fetcher.extract_depth == "basic"
        assert fetcher.include_images is False
        assert fetcher.extract_params is None
        assert fetcher.api_key.resolve_value() == "test-key"

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

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        fetcher = TavilyFetcher(extract_depth="advanced", extract_params={"format": "text"})
        data = component_to_dict(fetcher, "TavilyFetcher")
        assert data["type"] == "haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyFetcher"
        assert data["init_parameters"]["extract_depth"] == "advanced"
        assert data["init_parameters"]["extract_params"] == {"format": "text"}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
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

    def test_warm_up_initializes_clients(self):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        assert fetcher._tavily_client is None
        assert fetcher._async_tavily_client is None
        fetcher.warm_up()
        assert fetcher._tavily_client is not None
        assert fetcher._async_tavily_client is not None

    def test_run_triggers_warm_up(self, extract_response):
        with (
            patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyClient") as mock_cls,
            patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.AsyncTavilyClient"),
        ):
            mock_cls.return_value.extract.return_value = extract_response
            fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
            fetcher.run(urls=["https://example.com"])
            mock_cls.assert_called_once_with(api_key="test-key")

    def test_run_raises_runtime_error_when_warm_up_fails(self, monkeypatch):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(fetcher, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="TavilyFetcher client failed to initialize"):
            fetcher.run(urls=["https://example.com"])

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
        with (
            patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.TavilyClient"),
            patch("haystack_integrations.components.fetchers.tavily.tavily_fetcher.AsyncTavilyClient") as mock_cls,
        ):
            mock_cls.return_value.extract = AsyncMock(return_value=extract_response)
            fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
            await fetcher.run_async(urls=["https://example.com"])
            mock_cls.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_run_async_raises_runtime_error_when_warm_up_fails(self, monkeypatch):
        fetcher = TavilyFetcher(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(fetcher, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="TavilyFetcher async client failed to initialize"):
            await fetcher.run_async(urls=["https://example.com"])

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
    def test_run_integration(self):
        fetcher = TavilyFetcher(api_key=Secret.from_env_var("TAVILY_API_KEY"))
        result = fetcher.run(urls=["https://haystack.deepset.ai"])
        assert len(result["documents"]) > 0
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content
        assert result["meta"]["response_time"] is not None

    @pytest.mark.skipif(
        not os.environ.get("TAVILY_API_KEY"),
        reason="Export TAVILY_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        fetcher = TavilyFetcher(api_key=Secret.from_env_var("TAVILY_API_KEY"))
        result = await fetcher.run_async(urls=["https://haystack.deepset.ai"])
        assert len(result["documents"]) > 0
        assert result["meta"]["response_time"] is not None
