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

from haystack_integrations.components.websearch.outcrawl import OutcrawlWebSearch
from haystack_integrations.components.websearch.outcrawl.websearch import OUTCRAWL_SEARCH_URL

SAMPLE_RESPONSE = {
    "hits": [
        {"url": "https://example.com", "title": "Example Title", "snippet": "First snippet.", "rank": 1},
        {"url": "https://example.org", "title": None, "snippet": None, "rank": 2},
    ]
}


def _make_transport(captured: list[httpx.Request], response: dict | None = None, status_code: int = 200):
    payload = response if response is not None else SAMPLE_RESPONSE

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status_code=status_code, json=payload)

    return httpx.MockTransport(handler)


def _component_with_transport(
    captured: list[httpx.Request],
    response: dict | None = None,
    status_code: int = 200,
    **kwargs,
) -> OutcrawlWebSearch:
    component = OutcrawlWebSearch(api_key=Secret.from_token("test-api-key"), **kwargs)
    transport = _make_transport(captured, response, status_code)
    component._client = httpx.Client(transport=transport)
    component._async_client = httpx.AsyncClient(transport=transport)
    return component


class TestOutcrawlWebSearch:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OUTCRAWL_API_KEY", "test-key")
        ws = OutcrawlWebSearch()
        assert ws.top_k == 10
        assert ws.search_params is None
        assert ws.timeout == 30.0
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_parameters(self):
        ws = OutcrawlWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=3,
            search_params={"country": "us"},
            timeout=10.0,
        )
        assert ws.top_k == 3
        assert ws.search_params == {"country": "us"}
        assert ws.timeout == 10.0

    def test_to_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OUTCRAWL_API_KEY", "test-key")
        ws = OutcrawlWebSearch(top_k=5, search_params={"country": "us"})
        data = component_to_dict(ws, "OutcrawlWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.outcrawl.websearch.OutcrawlWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["search_params"] == {"country": "us"}
        assert data["init_parameters"]["api_key"] == {
            "env_vars": ["OUTCRAWL_API_KEY"],
            "strict": True,
            "type": "env_var",
        }

        restored = component_from_dict(OutcrawlWebSearch, data, "OutcrawlWebSearch")
        assert restored.top_k == 5
        assert restored.search_params == {"country": "us"}

    def test_warm_up_only_creates_the_sync_client(self):
        ws = OutcrawlWebSearch(api_key=Secret.from_token("test-key"))
        ws.warm_up()
        assert ws._client is not None
        assert ws._async_client is None
        ws.close()
        assert ws._client is None

    @pytest.mark.asyncio
    async def test_warm_up_async_and_close_async(self):
        ws = OutcrawlWebSearch(api_key=Secret.from_token("test-key"))
        await ws.warm_up_async()
        assert ws._async_client is not None
        await ws.close_async()
        assert ws._async_client is None

    def test_run_sends_query_limit_and_bearer_key(self):
        captured: list[httpx.Request] = []
        ws = _component_with_transport(captured, top_k=7)

        ws.run(query="haystack")

        assert len(captured) == 1
        request = captured[0]
        assert str(request.url) == OUTCRAWL_SEARCH_URL
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert json.loads(request.content) == {"query": "haystack", "limit": 7}

    def test_run_search_params_pass_through_and_none_values_are_dropped(self):
        captured: list[httpx.Request] = []
        ws = _component_with_transport(captured, top_k=2, search_params={"country": "us", "language": None})

        ws.run(query="haystack")

        assert json.loads(captured[0].content) == {"query": "haystack", "limit": 2, "country": "us"}

    def test_run_per_run_overrides(self):
        captured: list[httpx.Request] = []
        ws = _component_with_transport(captured, top_k=10, search_params={"country": "us"})

        ws.run(query="haystack", top_k=3, search_params={"country": "de"})

        assert json.loads(captured[0].content) == {"query": "haystack", "limit": 3, "country": "de"}

    def test_run_without_top_k_omits_limit(self):
        captured: list[httpx.Request] = []
        ws = _component_with_transport(captured, top_k=None)

        ws.run(query="haystack")

        assert json.loads(captured[0].content) == {"query": "haystack"}

    def test_run_parses_hits(self):
        captured: list[httpx.Request] = []
        ws = _component_with_transport(captured)

        result = ws.run(query="haystack")

        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert result["documents"][0].content == "First snippet."
        assert result["documents"][0].meta == {"title": "Example Title", "url": "https://example.com", "rank": 1}
        # Missing title and snippet fall back to empty strings.
        assert result["documents"][1].content == ""
        assert result["documents"][1].meta["title"] == ""
        assert result["links"] == ["https://example.com", "https://example.org"]

    def test_run_with_no_hits(self):
        ws = _component_with_transport([], response={"hits": []})
        assert ws.run(query="nothing") == {"documents": [], "links": []}

    def test_run_raises_on_http_error(self):
        ws = _component_with_transport([], response={"code": "unauthorized"}, status_code=401)
        with pytest.raises(httpx.HTTPStatusError):
            ws.run(query="haystack")

    @pytest.mark.asyncio
    async def test_run_async(self):
        captured: list[httpx.Request] = []
        ws = _component_with_transport(captured, top_k=4)

        result = await ws.run_async(query="haystack", top_k=2)

        assert json.loads(captured[0].content) == {"query": "haystack", "limit": 2}
        assert captured[0].headers["Authorization"] == "Bearer test-api-key"
        assert result["links"] == ["https://example.com", "https://example.org"]


@pytest.mark.skipif(
    not os.environ.get("OUTCRAWL_API_KEY", None),
    reason="Export OUTCRAWL_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestOutcrawlWebSearchIntegration:
    def test_live_run(self):
        ws = OutcrawlWebSearch(top_k=3)
        result = ws.run(query="Haystack open-source AI framework")
        assert result["documents"]
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(link.startswith("http") for link in result["links"])

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        ws = OutcrawlWebSearch(top_k=3)
        result = await ws.run_async(query="Haystack open-source AI framework")
        assert result["documents"]
