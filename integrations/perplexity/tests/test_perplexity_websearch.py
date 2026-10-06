# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.perplexity import PerplexityWebSearch
from haystack_integrations.components.websearch.perplexity.perplexity_websearch import (
    PERPLEXITY_SEARCH_URL,
)

SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Example Title",
            "url": "https://example.com",
            "snippet": "Example snippet content",
            "date": "2026-01-01",
            "last_updated": "2026-04-01",
        },
        {
            "title": "Second",
            "url": "https://example.org",
            "snippet": "Second snippet",
        },
    ],
    "id": "abc-123",
    "server_time": "2026-04-30T00:00:00Z",
}


def _make_transport(captured: list[httpx.Request], response: dict | None = None, status_code: int = 200):
    payload = response if response is not None else SAMPLE_RESPONSE

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler)


class TestInitialization:
    def test_init_default(self):
        ws = PerplexityWebSearch()
        assert ws.top_k == 10
        assert ws.search_params is None
        assert ws.timeout == 30.0
        assert ws.api_key == Secret.from_env_var("PERPLEXITY_API_KEY")
        assert ws._client is None
        assert ws._async_client is None

    def test_init_with_params(self):
        ws = PerplexityWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            search_params={"search_recency_filter": "week"},
            timeout=10.0,
        )
        assert ws.top_k == 5
        assert ws.search_params == {"search_recency_filter": "week"}
        assert ws.timeout == 10.0


class TestSerialization:
    def test_to_dict(self):
        ws = PerplexityWebSearch(top_k=5, search_params={"country": "US"})
        data = component_to_dict(ws, "PerplexityWebSearch")
        expected_type = "haystack_integrations.components.websearch.perplexity.perplexity_websearch.PerplexityWebSearch"
        assert data["type"] == expected_type
        assert data["init_parameters"]["api_key"] == Secret.from_env_var("PERPLEXITY_API_KEY").to_dict()
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["search_params"] == {"country": "US"}

    def test_from_dict(self):
        data = {
            "type": ("haystack_integrations.components.websearch.perplexity.perplexity_websearch.PerplexityWebSearch"),
            "init_parameters": {
                "top_k": 3,
                "search_params": {"search_recency_filter": "day"},
                "timeout": 15.0,
                "api_key": {
                    "env_vars": ["PERPLEXITY_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
            },
        }
        ws = component_from_dict(PerplexityWebSearch, data, "PerplexityWebSearch")
        assert ws.top_k == 3
        assert ws.search_params == {"search_recency_filter": "day"}
        assert ws.timeout == 15.0


class TestComponentLifecycle:
    @patch("haystack_integrations.components.websearch.perplexity.perplexity_websearch.httpx.Client")
    def test_sync_lifecycle(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"))

        ws.close()
        ws.warm_up()
        ws.warm_up()
        assert ws._client is client
        assert ws._async_client is None
        mock_client_cls.assert_called_once_with(timeout=30.0)

        ws.close()
        client.close.assert_called_once_with()
        assert ws._client is None

        ws.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.websearch.perplexity.perplexity_websearch.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, mock_client_cls):
        client = MagicMock(aclose=AsyncMock())
        mock_client_cls.return_value = client
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"))

        await ws.close_async()
        await ws.warm_up_async()
        await ws.warm_up_async()
        assert ws._async_client is client
        assert ws._client is None
        mock_client_cls.assert_called_once_with(timeout=30.0)

        await ws.close_async()
        client.aclose.assert_awaited_once_with()
        assert ws._async_client is None

        await ws.warm_up_async()
        assert mock_client_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        sync_client = MagicMock()
        async_client = MagicMock(aclose=AsyncMock())
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"))
        ws._client = sync_client
        ws._async_client = async_client

        ws.close()
        assert ws._client is None
        assert ws._async_client is async_client
        async_client.aclose.assert_not_awaited()

        await ws.close_async()
        assert ws._async_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    def test_run_returns_documents_and_links(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._client = httpx.Client(transport=_make_transport(captured))

        result = ws.run(query="test query")

        assert len(result["documents"]) == 2
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example snippet content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["documents"][0].meta["date"] == "2026-01-01"
        assert result["documents"][0].meta["last_updated"] == "2026-04-01"
        assert result["links"] == ["https://example.com", "https://example.org"]

        assert len(captured) == 1
        request = captured[0]
        assert str(request.url) == PERPLEXITY_SEARCH_URL
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body == {"query": "test query", "max_results": 10}

    def test_run_sends_attribution_and_auth_headers(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(api_key=Secret.from_token("secret-key"))
        ws._client = httpx.Client(transport=_make_transport(captured))

        ws.run(query="test")

        request = captured[0]
        assert request.headers["Authorization"] == "Bearer secret-key"
        assert request.headers["Content-Type"] == "application/json"
        attribution = request.headers["X-Pplx-Integration"]
        assert attribution.startswith("haystack/")

    def test_run_overrides_params_at_runtime(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"country": "US"},
        )
        ws._client = httpx.Client(transport=_make_transport(captured))

        ws.run(query="test", search_params={"country": "DE", "max_results": 3})

        body = json.loads(captured[0].content)
        assert body == {"query": "test", "country": "DE", "max_results": 3}

    def test_run_uses_init_search_params_when_no_runtime_override(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=5,
            search_params={"search_recency_filter": "week"},
        )
        ws._client = httpx.Client(transport=_make_transport(captured))

        ws.run(query="hello")

        body = json.loads(captured[0].content)
        assert body == {
            "query": "hello",
            "search_recency_filter": "week",
            "max_results": 5,
        }

    @pytest.mark.asyncio
    async def test_run_async(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._async_client = httpx.AsyncClient(transport=_make_transport(captured))

        result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 2
        assert result["links"] == ["https://example.com", "https://example.org"]
        assert len(captured) == 1
        assert captured[0].headers["X-Pplx-Integration"].startswith("haystack/")

    def test_run_raises_on_http_error(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"))
        ws._client = httpx.Client(
            transport=_make_transport(captured, response={"error": "bad request"}, status_code=400)
        )

        with pytest.raises(httpx.HTTPStatusError):
            ws.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async_raises_on_http_error(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"))
        ws._async_client = httpx.AsyncClient(
            transport=_make_transport(captured, response={"error": "bad request"}, status_code=500)
        )

        with pytest.raises(httpx.HTTPStatusError):
            await ws.run_async(query="test")

    def test_run_empty_results(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-key"))
        ws._client = httpx.Client(transport=_make_transport(captured, response={"results": []}))

        result = ws.run(query="obscure query")

        assert result["documents"] == []
        assert result["links"] == []

    def test_run_drops_none_valued_search_params(self):
        captured: list[httpx.Request] = []
        ws = PerplexityWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=5,
            search_params={"country": None, "search_recency_filter": "month"},
        )
        ws._client = httpx.Client(transport=_make_transport(captured))

        ws.run(query="hi")

        body = json.loads(captured[0].content)
        assert "country" not in body
        assert body["search_recency_filter"] == "month"


@pytest.mark.skipif(
    not os.environ.get("PERPLEXITY_API_KEY"),
    reason="Export PERPLEXITY_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestIntegration:
    def test_run_integration(self):
        ws = PerplexityWebSearch(api_key=Secret.from_env_var("PERPLEXITY_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)

    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ws = PerplexityWebSearch(api_key=Secret.from_env_var("PERPLEXITY_API_KEY"), top_k=3)
        result = await ws.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
