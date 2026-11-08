# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.fetchers.outcrawl import OutcrawlFetcher
from haystack_integrations.components.fetchers.outcrawl.fetcher import OUTCRAWL_SCRAPE_URL

PAGE = {
    "url": "https://example.com/",
    "statusCode": 200,
    "markdown": "# Example\n\nHello.",
    "metadata": {"title": "Example Domain", "language": "en", "servedVia": "direct"},
}


def _component_with_transport(handler, **kwargs) -> OutcrawlFetcher:
    component = OutcrawlFetcher(api_key=Secret.from_token("test-api-key"), **kwargs)
    transport = httpx.MockTransport(handler)
    component._client = httpx.Client(transport=transport)
    component._async_client = httpx.AsyncClient(transport=transport)
    return component


def _ok_handler(captured: list[httpx.Request], payload=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=payload if payload is not None else PAGE)

    return handler


class TestOutcrawlFetcher:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OUTCRAWL_API_KEY", "test-key")
        fetcher = OutcrawlFetcher()
        assert fetcher.params is None
        assert fetcher._params == {"formats": ["markdown"]}
        assert fetcher.timeout == 65.0
        assert fetcher.api_key.resolve_value() == "test-key"

    def test_init_params_are_copied_and_default_format_is_kept(self):
        params = {"onlyMainContent": True}
        fetcher = OutcrawlFetcher(api_key=Secret.from_token("key"), params=params)
        assert fetcher._params == {"onlyMainContent": True, "formats": ["markdown"]}
        assert params == {"onlyMainContent": True}

    def test_to_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OUTCRAWL_API_KEY", "test-key")
        fetcher = OutcrawlFetcher(params={"onlyMainContent": True}, timeout=20.0)
        data = component_to_dict(fetcher, "OutcrawlFetcher")
        assert data["type"] == "haystack_integrations.components.fetchers.outcrawl.fetcher.OutcrawlFetcher"
        assert data["init_parameters"]["params"] == {"onlyMainContent": True}
        assert data["init_parameters"]["timeout"] == 20.0

        restored = component_from_dict(OutcrawlFetcher, data, "OutcrawlFetcher")
        assert restored.params == {"onlyMainContent": True}
        assert restored.timeout == 20.0

    def test_warm_up_and_close(self):
        fetcher = OutcrawlFetcher(api_key=Secret.from_token("test-key"))
        fetcher.warm_up()
        assert fetcher._client is not None
        assert fetcher._async_client is None
        fetcher.close()
        assert fetcher._client is None

    @pytest.mark.asyncio
    async def test_warm_up_async_and_close_async(self):
        fetcher = OutcrawlFetcher(api_key=Secret.from_token("test-key"))
        await fetcher.warm_up_async()
        assert fetcher._async_client is not None
        await fetcher.close_async()
        assert fetcher._async_client is None

    def test_run_sends_url_format_and_bearer_key(self):
        captured: list[httpx.Request] = []
        fetcher = _component_with_transport(_ok_handler(captured))

        fetcher.run(urls=["https://example.com"])

        assert len(captured) == 1
        assert str(captured[0].url) == OUTCRAWL_SCRAPE_URL
        assert captured[0].headers["Authorization"] == "Bearer test-api-key"
        assert json.loads(captured[0].content) == {"url": "https://example.com", "formats": ["markdown"]}

    def test_run_returns_documents(self):
        fetcher = _component_with_transport(_ok_handler([]))

        result = fetcher.run(urls=["https://example.com"])

        assert len(result["documents"]) == 1
        document = result["documents"][0]
        assert isinstance(document, Document)
        assert document.content == "# Example\n\nHello."
        assert document.meta == {
            "url": "https://example.com/",
            "title": "Example Domain",
            "language": "en",
            "status_code": 200,
        }

    def test_run_per_run_params_replace_init_params(self):
        captured: list[httpx.Request] = []
        fetcher = _component_with_transport(_ok_handler(captured), params={"onlyMainContent": True})

        fetcher.run(urls=["https://example.com"], params={"formats": ["markdown", "html"]})

        assert json.loads(captured[0].content) == {"url": "https://example.com", "formats": ["markdown", "html"]}

    def test_run_skips_failed_and_empty_pages(self):
        def handler(request: httpx.Request) -> httpx.Response:
            url = json.loads(request.content)["url"]
            if url.endswith("/error"):
                return httpx.Response(502, json={"code": "upstream_error"})
            if url.endswith("/empty"):
                return httpx.Response(200, json={"url": url, "markdown": "   "})
            return httpx.Response(200, json={**PAGE, "url": url})

        fetcher = _component_with_transport(handler)

        result = fetcher.run(urls=["https://example.com/error", "https://example.com/empty", "https://example.com/ok"])

        assert [doc.meta["url"] for doc in result["documents"]] == ["https://example.com/ok"]

    def test_run_handles_non_object_response(self):
        fetcher = _component_with_transport(_ok_handler([], payload=["not", "an", "object"]))
        assert fetcher.run(urls=["https://example.com"]) == {"documents": []}

    def test_run_falls_back_to_requested_url(self):
        fetcher = _component_with_transport(_ok_handler([], payload={"markdown": "text"}))
        result = fetcher.run(urls=["https://example.com/page"])
        assert result["documents"][0].meta == {"url": "https://example.com/page"}

    @pytest.mark.asyncio
    async def test_run_async(self):
        captured: list[httpx.Request] = []
        fetcher = _component_with_transport(_ok_handler(captured))

        result = await fetcher.run_async(urls=["https://example.com"])

        assert json.loads(captured[0].content) == {"url": "https://example.com", "formats": ["markdown"]}
        assert result["documents"][0].content == "# Example\n\nHello."

    @pytest.mark.asyncio
    async def test_run_async_skips_failed_pages(self):
        fetcher = _component_with_transport(lambda _request: httpx.Response(500, json={}))
        assert await fetcher.run_async(urls=["https://example.com"]) == {"documents": []}


@pytest.mark.skipif(
    not os.environ.get("OUTCRAWL_API_KEY", None),
    reason="Export OUTCRAWL_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestOutcrawlFetcherIntegration:
    def test_live_run(self):
        result = OutcrawlFetcher().run(urls=["https://example.com"])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        result = await OutcrawlFetcher().run_async(urls=["https://example.com"])
        assert len(result["documents"]) == 1
