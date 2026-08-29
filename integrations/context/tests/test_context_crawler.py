# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.fetchers.context import ContextCrawler

CRAWL_RESPONSE = {
    "results": [
        {
            "markdown": "# Haystack",
            "metadata": {
                "url": "https://haystack.deepset.ai",
                "title": "Haystack",
                "crawlDepth": 0,
                "statusCode": 200,
                "success": True,
            },
        },
        {
            "markdown": "",
            "metadata": {
                "url": "https://haystack.deepset.ai/missing",
                "title": "",
                "crawlDepth": 1,
                "statusCode": 404,
                "success": False,
            },
        },
    ],
    "metadata": {"numUrls": 2, "numSucceeded": 1, "numFailed": 1},
}


class TestContextCrawler:
    def test_init_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_API_KEY", "test-key")
        crawler = ContextCrawler()

        assert crawler.api_key.resolve_value() == "test-key"
        assert crawler.crawl_params is None
        assert crawler.timeout == 120

    def test_serialization_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_API_KEY", "test-key")
        crawler = ContextCrawler(crawl_params={"maxPages": 5, "maxDepth": 2}, timeout=90)

        data = component_to_dict(crawler, name="crawler")
        restored = component_from_dict(ContextCrawler, data, name="crawler")

        assert restored.crawl_params == {"maxPages": 5, "maxDepth": 2}
        assert restored.timeout == 90
        assert restored.api_key.resolve_value() == "test-key"

    @patch("haystack_integrations.components.fetchers.context.context_crawler.request_context")
    def test_run_returns_only_successful_pages(self, request_context) -> None:
        request_context.return_value = CRAWL_RESPONSE
        crawl_params = {"maxPages": 5, "maxDepth": 2}
        original_params = deepcopy(crawl_params)
        crawler = ContextCrawler(api_key=Secret.from_token("test-key"), crawl_params=crawl_params)

        result = crawler.run(urls=["https://haystack.deepset.ai"])

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "# Haystack"
        assert result["documents"][0].meta["success"] is True
        assert request_context.call_args.kwargs["json"] == {
            "url": "https://haystack.deepset.ai",
            "maxPages": 5,
            "maxDepth": 2,
            "useMainContentOnly": True,
            "includeLinks": True,
            "includeImages": False,
        }
        assert crawl_params == original_params

    @patch("haystack_integrations.components.fetchers.context.context_crawler.request_context")
    def test_run_defaults_to_one_page(self, request_context) -> None:
        request_context.return_value = CRAWL_RESPONSE
        crawler = ContextCrawler(api_key=Secret.from_token("test-key"))

        crawler.run(urls=["https://haystack.deepset.ai"])

        assert request_context.call_args.kwargs["json"]["maxPages"] == 1

    @patch("haystack_integrations.components.fetchers.context.context_crawler.request_context_async")
    @pytest.mark.asyncio
    async def test_run_async_crawls_concurrently(self, request_context) -> None:
        request_context.side_effect = AsyncMock(return_value=CRAWL_RESPONSE)
        crawler = ContextCrawler(api_key=Secret.from_token("test-key"))

        result = await crawler.run_async(urls=["https://haystack.deepset.ai", "https://docs.haystack.deepset.ai"])

        assert len(result["documents"]) == 2
        assert request_context.await_count == 2

    def test_parse_response_handles_invalid_results(self) -> None:
        assert ContextCrawler._documents_from_response({"results": None}) == []
        assert ContextCrawler._documents_from_response({"results": [None, "invalid"]}) == []

    @pytest.mark.skipif(
        not os.environ.get("CONTEXT_API_KEY"),
        reason="Export CONTEXT_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self) -> None:
        crawler = ContextCrawler(crawl_params={"maxPages": 1, "maxDepth": 0})
        result = crawler.run(urls=["https://haystack.deepset.ai"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content
        assert result["documents"][0].meta["success"] is True

    @pytest.mark.skipif(
        not os.environ.get("CONTEXT_API_KEY"),
        reason="Export CONTEXT_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self) -> None:
        crawler = ContextCrawler(crawl_params={"maxPages": 1, "maxDepth": 0})
        result = await crawler.run_async(urls=["https://haystack.deepset.ai"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content
        assert result["documents"][0].meta["success"] is True
