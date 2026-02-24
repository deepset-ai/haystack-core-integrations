# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from firecrawl.v2.utils.error_handler import UnauthorizedError
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.fetchers.firecrawl import FirecrawlCrawler


class TestFirecrawlCrawler:
    @pytest.fixture
    def page(self) -> MagicMock:
        page = MagicMock()
        page.markdown = "Page content"
        page.metadata_dict = {"url": "https://example.com"}
        return page

    @pytest.fixture
    def crawl_response(self, page) -> MagicMock:
        crawl_response = MagicMock()
        crawl_response.status = "completed"
        crawl_response.data = [page]
        return crawl_response

    @pytest.fixture
    def mock_client(self, crawl_response) -> MagicMock:
        mock_client = MagicMock()
        mock_client.crawl.return_value = crawl_response
        return mock_client

    @pytest.fixture
    def mock_async_client(self, crawl_response) -> MagicMock:
        mock_async_client = MagicMock()
        mock_async_client.crawl = AsyncMock(return_value=crawl_response)
        return mock_async_client

    def test_init_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        fetcher = FirecrawlCrawler()
        assert fetcher._params == {"limit": 1, "scrape_options": {"formats": ["markdown"]}}
        assert fetcher.api_key.resolve_value() == "test-key"

    def test_init_with_params(self) -> None:
        fetcher = FirecrawlCrawler(
            api_key=Secret.from_token("custom-key"),
            params={"limit": 10},
        )
        assert fetcher._params == {"limit": 10, "scrape_options": {"formats": ["markdown"]}}
        assert fetcher.api_key.resolve_value() == "custom-key"

    def test_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        fetcher = FirecrawlCrawler(
            api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
            params={"limit": 5},
        )
        data = component_to_dict(fetcher, "FirecrawlFetcher")
        assert data["type"] == "haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler.FirecrawlCrawler"
        assert data["init_parameters"]["params"] == {"limit": 5}
        assert "api_key" in data["init_parameters"]

    def test_from_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler.FirecrawlCrawler",
            "init_parameters": {
                "params": {"limit": 3},
                "api_key": {"env_vars": ["FIRECRAWL_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        fetcher = component_from_dict(FirecrawlCrawler, data, "FirecrawlFetcher")
        assert fetcher.params == {"limit": 3}
        assert fetcher.api_key.resolve_value() == "test-key"

    def test_run_with_mocked_firecrawl_client(self, mock_client) -> None:
        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))

        fetcher._firecrawl_client = mock_client

        result = fetcher.run(urls=["https://example.com", "https://docs.example.com"])

        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert result["documents"][0].content == "Page content"
        assert result["documents"][0].meta["url"] == "https://example.com"

        assert mock_client.crawl.call_count == 2
        mock_client.crawl.assert_any_call(
            url="https://example.com",
            limit=1,
            scrape_options={"formats": ["markdown"]},
        )
        mock_client.crawl.assert_any_call(
            url="https://docs.example.com",
            limit=1,
            scrape_options={"formats": ["markdown"]},
        )

    def test_run_overrides_init_params_with_runtime_params(self, mock_client) -> None:
        fetcher = FirecrawlCrawler(
            api_key=Secret.from_token("test-key"),
            params={"limit": 1, "scrape_options": {"formats": ["markdown"]}},
        )

        fetcher._firecrawl_client = mock_client

        fetcher.run(urls=["https://example.com"], params={"limit": 5})

        mock_client.crawl.assert_called_once_with(
            url="https://example.com",
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_run_async_uses_async_firecrawl_client(self, mock_async_client) -> None:
        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))

        fetcher._async_firecrawl_client = mock_async_client

        result = await fetcher.run_async(urls=["https://example.com"], params={"limit": 2})

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Page content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        mock_async_client.crawl.assert_awaited_once_with(
            url="https://example.com",
            limit=2,
        )

    def test_run_calls_warm_up(self, crawl_response) -> None:
        with (
            patch(
                "haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler.Firecrawl"
            ) as mock_firecrawl_client,
            patch("haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler.AsyncFirecrawl"),
        ):
            mock_firecrawl_client.return_value.crawl.return_value = crawl_response

            fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
            fetcher.run(urls=["https://example.com"])

            assert fetcher._firecrawl_client is mock_firecrawl_client.return_value
            mock_firecrawl_client.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_run_async_calls_warm_up(self, crawl_response) -> None:
        with (
            patch("haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler.Firecrawl"),
            patch(
                "haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler.AsyncFirecrawl"
            ) as mock_async_firecrawl_client,
        ):
            mock_async_firecrawl_client.return_value.crawl = AsyncMock(return_value=crawl_response)

            fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
            await fetcher.run_async(urls=["https://example.com"])

            assert fetcher._async_firecrawl_client is mock_async_firecrawl_client.return_value
            mock_async_firecrawl_client.assert_called_once_with(api_key="test-key")

    def test_run_returns_empty_on_crawl_error(self, mock_client) -> None:
        mock_client.crawl.side_effect = UnauthorizedError(
            "Unauthorized: Failed to start crawl. Unauthorized: Invalid token - No additional error details provided."
        )

        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
        fetcher._firecrawl_client = mock_client

        result = fetcher.run(urls=["https://example.com"])

        assert result["documents"] == []
        mock_client.crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_returns_empty_on_crawl_error(self, mock_async_client) -> None:
        mock_async_client.crawl = AsyncMock(
            side_effect=UnauthorizedError(
                "Unauthorized: Failed to start crawl. Unauthorized: Invalid token"
                " - No additional error details provided."
            )
        )

        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
        fetcher._async_firecrawl_client = mock_async_client

        result = await fetcher.run_async(urls=["https://example.com"])

        assert result["documents"] == []
        mock_async_client.crawl.assert_awaited_once()

    def test_run_logs_error_on_failed_crawl_status(self, mock_client, page, caplog) -> None:
        failed_response = MagicMock()
        failed_response.status = "failed"
        failed_response.data = [page]
        mock_client.crawl.return_value = failed_response

        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
        fetcher._firecrawl_client = mock_client

        fetcher.run(urls=["https://example.com"])

        assert "Failed to crawl website https://example.com: failed" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_logs_error_on_failed_crawl_status(self, mock_async_client, page, caplog) -> None:
        failed_response = MagicMock()
        failed_response.status = "failed"
        failed_response.data = [page]
        mock_async_client.crawl = AsyncMock(return_value=failed_response)

        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
        fetcher._async_firecrawl_client = mock_async_client

        await fetcher.run_async(urls=["https://example.com"])

        assert "Failed to crawl website https://example.com: failed" in caplog.text

    def test_warm_up_initializes_clients(self) -> None:
        fetcher = FirecrawlCrawler(api_key=Secret.from_token("test-key"))
        assert fetcher._firecrawl_client is None
        assert fetcher._async_firecrawl_client is None

        fetcher.warm_up()

        assert fetcher._firecrawl_client is not None
        assert fetcher._async_firecrawl_client is not None

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"),
        reason="Export FIRECRAWL_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self) -> None:
        fetcher = FirecrawlCrawler(api_key=Secret.from_env_var("FIRECRAWL_API_KEY"), params={"limit": 1})
        result = fetcher.run(urls=["https://docs.haystack.deepset.ai/docs/intro"], params={"limit": 1})

        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert "Haystack" in result["documents"][0].content
        assert "https://docs.haystack.deepset.ai/docs/intro" == result["documents"][0].meta["url"]

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"),
        reason="Export FIRECRAWL_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self) -> None:
        fetcher = FirecrawlCrawler(api_key=Secret.from_env_var("FIRECRAWL_API_KEY"), params={"limit": 1})
        result = await fetcher.run_async(urls=["https://docs.haystack.deepset.ai/docs/intro"])

        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert "Haystack" in result["documents"][0].content
        assert "https://docs.haystack.deepset.ai/docs/intro" == result["documents"][0].meta["url"]
