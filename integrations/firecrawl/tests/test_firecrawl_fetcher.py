# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.fetchers.firecrawl import FirecrawlFetcher


class TestFirecrawlFetcher:
    def test_init_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        fetcher = FirecrawlFetcher()
        assert fetcher._params == {"limit": 1, "scrape_options": {"formats": ["markdown"]}}
        assert fetcher.api_key.resolve_value() == "test-key"

    def test_init_with_params(self) -> None:
        fetcher = FirecrawlFetcher(
            api_key=Secret.from_token("custom-key"),
            params={"limit": 10},
        )
        assert fetcher._params == {"limit": 10, "scrape_options": {"formats": ["markdown"]}}
        assert fetcher.api_key.resolve_value() == "custom-key"

    def test_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        fetcher = FirecrawlFetcher(
            api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
            params={"limit": 5},
        )
        data = fetcher.to_dict()
        assert data["type"] == "haystack_integrations.components.fetchers.firecrawl.firecrawl_fetcher.FirecrawlFetcher"
        assert data["init_parameters"]["params"] == {"limit": 5}
        assert "api_key" in data["init_parameters"]

    def test_from_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.fetchers.firecrawl.firecrawl_fetcher.FirecrawlFetcher",
            "init_parameters": {
                "params": {"limit": 3},
                "api_key": {"env_vars": ["FIRECRAWL_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        fetcher = FirecrawlFetcher.from_dict(data)
        assert fetcher.params == {"limit": 3}
        assert fetcher.api_key.resolve_value() == "test-key"

    def test_run_with_mocked_firecrawl_client(self) -> None:
        fetcher = FirecrawlFetcher(api_key=Secret.from_token("test-key"))

        page = MagicMock()
        page.markdown = "Page content"
        page.metadata_dict = {"url": "https://example.com"}

        crawl_response = MagicMock()
        crawl_response.status = "completed"
        crawl_response.data = [page]

        mock_client = MagicMock()
        mock_client.crawl.return_value = crawl_response
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

    def test_run_merges_runtime_params_before_client_call(self) -> None:
        fetcher = FirecrawlFetcher(
            api_key=Secret.from_token("test-key"),
            params={"limit": 1, "scrape_options": {"formats": ["markdown"]}},
        )

        crawl_response = MagicMock()
        crawl_response.status = "completed"
        crawl_response.data = []

        mock_client = MagicMock()
        mock_client.crawl.return_value = crawl_response
        fetcher._firecrawl_client = mock_client

        fetcher.run(urls=["https://example.com"], params={"limit": 5})

        mock_client.crawl.assert_called_once_with(
            url="https://example.com",
            limit=5,
            scrape_options={"formats": ["markdown"]},
        )

    def test_run_calls_warm_up_when_client_not_initialized(self) -> None:
        fetcher = FirecrawlFetcher(api_key=Secret.from_token("test-key"))

        crawl_response = MagicMock()
        crawl_response.status = "completed"
        crawl_response.data = []

        mock_client = MagicMock()
        mock_client.crawl.return_value = crawl_response

        with patch.object(fetcher, "warm_up") as warm_up_mock:
            warm_up_mock.side_effect = lambda: setattr(fetcher, "_firecrawl_client", mock_client)
            fetcher.run(urls=["https://example.com"])

        warm_up_mock.assert_called_once()
        mock_client.crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_uses_async_firecrawl_client(self) -> None:
        fetcher = FirecrawlFetcher(api_key=Secret.from_token("test-key"))

        page = MagicMock()
        page.markdown = "Page content"
        page.metadata_dict = {"url": "https://example.com"}

        crawl_response = MagicMock()
        crawl_response.status = "completed"
        crawl_response.data = [page]

        mock_async_client = MagicMock()
        mock_async_client.crawl = AsyncMock(return_value=crawl_response)
        fetcher._async_firecrawl_client = mock_async_client

        result = await fetcher.run_async(urls=["https://example.com"], params={"limit": 2})

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Page content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        mock_async_client.crawl.assert_awaited_once_with(
            url="https://example.com",
            limit=2,
            scrape_options={"formats": ["markdown"]},
        )

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"),
        reason="Export FIRECRAWL_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self) -> None:
        fetcher = FirecrawlFetcher(api_key=Secret.from_env_var("FIRECRAWL_API_KEY"), params={"limit": 1})
        result = fetcher.run(urls=["https://docs.haystack.deepset.ai/docs/intro"])

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
        fetcher = FirecrawlFetcher(api_key=Secret.from_env_var("FIRECRAWL_API_KEY"), params={"limit": 1})
        result = await fetcher.run_async(urls=["https://docs.haystack.deepset.ai/docs/intro"])

        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert "Haystack" in result["documents"][0].content
        assert "https://docs.haystack.deepset.ai/docs/intro" == result["documents"][0].meta["url"]
