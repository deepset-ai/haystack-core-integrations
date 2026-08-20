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

from haystack_integrations.components.websearch.parallel import ParallelWebSearch
from haystack_integrations.components.websearch.parallel.parallel_websearch import (
    PARALLEL_SEARCH_URL,
)

SAMPLE_RESPONSE = {
    "search_id": "search_abc123",
    "session_id": "session_xyz",
    "results": [
        {
            "url": "https://example.com",
            "title": "Example Title",
            "publish_date": "2026-01-01",
            "excerpts": ["First excerpt.", "Second excerpt."],
        },
        {
            "url": "https://example.org",
            "title": None,
            "publish_date": None,
            "excerpts": ["Only excerpt."],
        },
    ],
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
) -> ParallelWebSearch:
    component = ParallelWebSearch(api_key=Secret.from_token("test-api-key"), **kwargs)
    component.warm_up()
    transport = _make_transport(captured, response, status_code)
    component._client = httpx.Client(transport=transport)
    component._async_client = httpx.AsyncClient(transport=transport)
    return component


class TestParallelWebSearch:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "test-api-key")
        component = ParallelWebSearch()
        assert component.api_key == Secret.from_env_var("PARALLEL_API_KEY")
        assert component.top_k == 10
        assert component.search_params is None
        assert component.timeout == 30.0

    def test_init_with_parameters(self):
        component = ParallelWebSearch(
            api_key=Secret.from_token("test-api-key"),
            top_k=5,
            search_params={"mode": "turbo"},
            timeout=10.0,
        )
        assert component.top_k == 5
        assert component.search_params == {"mode": "turbo"}
        assert component.timeout == 10.0

    def test_to_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "test-api-key")
        component = ParallelWebSearch(top_k=7, search_params={"mode": "advanced"})
        data = component_to_dict(component, name="websearch")
        deserialized = component_from_dict(ParallelWebSearch, data, name="websearch")
        assert deserialized.top_k == 7
        assert deserialized.search_params == {"mode": "advanced"}

    def test_run_sends_query_objective_and_top_k(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(captured, top_k=5)

        component.run(query="What is Haystack?")

        assert len(captured) == 1
        request = captured[0]
        assert str(request.url) == PARALLEL_SEARCH_URL
        assert request.headers["x-api-key"] == "test-api-key"
        assert request.headers["x-parallel-integration"].startswith("haystack/")
        body = json.loads(request.content)
        assert body["search_queries"] == ["What is Haystack?"]
        assert body["objective"] == "What is Haystack?"
        assert body["advanced_settings"] == {"max_results": 5}

    def test_run_search_params_pass_through(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(
            captured,
            top_k=5,
            search_params={
                "mode": "advanced",
                "objective": "find peer-reviewed research",
                "max_chars_total": 4000,
                "advanced_settings": {
                    "source_policy": {"include_domains": ["arxiv.org"], "after_date": "2026-01-01"},
                    "fetch_policy": {"max_age_seconds": 600},
                },
            },
        )

        component.run(query="quantum computing")

        body = json.loads(captured[0].content)
        assert body["mode"] == "advanced"
        assert body["objective"] == "find peer-reviewed research"
        assert body["max_chars_total"] == 4000
        advanced_settings = body["advanced_settings"]
        assert advanced_settings["max_results"] == 5
        assert advanced_settings["source_policy"] == {"include_domains": ["arxiv.org"], "after_date": "2026-01-01"}
        assert advanced_settings["fetch_policy"] == {"max_age_seconds": 600}

    def test_run_explicit_max_results_wins_over_top_k(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(
            captured,
            top_k=5,
            search_params={"advanced_settings": {"max_results": 3}},
        )

        component.run(query="test")

        body = json.loads(captured[0].content)
        assert body["advanced_settings"]["max_results"] == 3

    def test_run_per_run_search_params_replace_init_params(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(captured, search_params={"mode": "turbo"})

        component.run(query="test", search_params={"mode": "basic"})

        body = json.loads(captured[0].content)
        assert body["mode"] == "basic"

    def test_run_parses_results(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(captured)

        result = component.run(query="test")

        documents = result["documents"]
        links = result["links"]
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert documents[0].content == "First excerpt. ... Second excerpt."
        assert documents[0].meta["title"] == "Example Title"
        assert documents[0].meta["url"] == "https://example.com"
        assert documents[0].meta["publish_date"] == "2026-01-01"
        assert documents[0].meta["excerpts"] == ["First excerpt.", "Second excerpt."]
        assert documents[1].content == "Only excerpt."
        assert documents[1].meta["title"] == ""
        assert "publish_date" not in documents[1].meta
        assert links == ["https://example.com", "https://example.org"]

    def test_run_raises_on_http_error(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(captured, response={"error": "unauthorized"}, status_code=401)

        with pytest.raises(httpx.HTTPStatusError):
            component.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async(self):
        captured: list[httpx.Request] = []
        component = _component_with_transport(captured, top_k=5)

        result = await component.run_async(query="What is Haystack?")

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["search_queries"] == ["What is Haystack?"]
        assert len(result["documents"]) == 2


@pytest.mark.skipif(
    not os.environ.get("PARALLEL_API_KEY"),
    reason="Export PARALLEL_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestParallelWebSearchInference:
    def test_live_run(self):
        component = ParallelWebSearch(top_k=3)
        result = component.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert result["documents"][0].content

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        component = ParallelWebSearch(top_k=3)
        result = await component.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
