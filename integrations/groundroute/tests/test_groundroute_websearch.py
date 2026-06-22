# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from httpx import ConnectTimeout, HTTPStatusError, Request, RequestError, Response

from haystack_integrations.components.websearch.groundroute import GroundRouteWebSearch
from haystack_integrations.components.websearch.groundroute.groundroute_websearch import GroundRouteError

MODULE = "haystack_integrations.components.websearch.groundroute.groundroute_websearch"

EXAMPLE_GROUNDROUTE_RESPONSE = {
    "request_id": "req-1",
    "results": [
        {
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "snippet": f"Snippet {i}",
            "source_engine": "serper",
        }
        for i in range(9)
    ],
    "answer": None,
    "citations": [],
    "degraded": False,
    "cache_meta": {"cache_tier": "miss"},
    "usage_meta": {"cost_usd": 0.001},
}


@pytest.fixture
def mock_groundroute_search_result() -> Generator[MagicMock, None, None]:
    with patch(f"{MODULE}.httpx") as mock_run:
        mock_run.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_GROUNDROUTE_RESPONSE)
        yield mock_run


@pytest.fixture
def mock_groundroute_search_result_async() -> Generator[MagicMock, None, None]:
    with patch(f"{MODULE}.httpx.AsyncClient") as mock_run:
        mock_client = AsyncMock()
        mock_client.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_GROUNDROUTE_RESPONSE)
        mock_client.__aenter__.return_value = mock_client
        mock_run.return_value = mock_client
        yield mock_run


class TestGroundRouteWebSearch:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("GROUNDROUTE_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            GroundRouteWebSearch()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GROUNDROUTE_API_KEY", "test-api-key")
        component = GroundRouteWebSearch(top_k=10, allowed_domains=["test.com"], search_params={"mode": "web"})
        data = component.to_dict()
        assert data == {
            "type": f"{MODULE}.GroundRouteWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["GROUNDROUTE_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 10,
                "allowed_domains": ["test.com"],
                "search_params": {"mode": "web"},
            },
        }

    def test_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("GROUNDROUTE_API_KEY", "test-api-key")
        component = GroundRouteWebSearch(top_k=7, allowed_domains=["test.com"], search_params={"mode": "web"})
        restored = GroundRouteWebSearch.from_dict(component.to_dict())
        assert restored.top_k == 7
        assert restored.allowed_domains == ["test.com"]
        assert restored.search_params == {"mode": "web"}
        assert restored.api_key == Secret.from_env_var("GROUNDROUTE_API_KEY")

    @pytest.mark.parametrize("top_k", [1, 5, 7])
    def test_web_search_top_k(self, mock_groundroute_search_result, top_k):
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = ws.run(query="What is a vector database?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)
        assert mock_groundroute_search_result.post.called

    @pytest.mark.parametrize("top_k", [1, 5, 7])
    @pytest.mark.asyncio
    async def test_web_search_top_k_async(self, mock_groundroute_search_result_async, top_k):
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = await ws.run_async(query="What is a vector database?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)
        assert mock_groundroute_search_result_async.called

    def test_maps_source_engine_into_meta(self, mock_groundroute_search_result):
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"), top_k=3)
        documents = ws.run(query="q")["documents"]
        assert all(doc.meta["source_engine"] == "serper" for doc in documents)
        assert all(doc.meta["link"].startswith("https://example.com/") for doc in documents)
        assert mock_groundroute_search_result.post.called

    def test_sends_bearer_auth_and_query(self, mock_groundroute_search_result):
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"), top_k=5, allowed_domains=["example.com"])
        ws.run(query="hello world")
        call = mock_groundroute_search_result.post.call_args
        assert call.kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert call.kwargs["json"]["query"] == "hello world"
        assert call.kwargs["json"]["max_results"] == 5
        assert call.kwargs["json"]["domains"] == ["example.com"]

    def test_answer_is_surfaced_as_lead_document(self):
        response = {
            "results": [{"title": "T", "url": "https://ex.com/a", "snippet": "s", "source_engine": "perplexity"}],
            "answer": "A synthesized answer.",
            "citations": ["https://ex.com/a"],
        }
        with patch(f"{MODULE}.httpx") as mock_run:
            mock_run.post.return_value = Mock(status_code=200, json=lambda: response)
            ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"), top_k=10)
            documents = ws.run(query="q")["documents"]
        assert documents[0].content == "A synthesized answer."
        assert documents[0].meta["citations"] == ["https://ex.com/a"]

    @patch("httpx.post")
    def test_timeout_error(self, mock_post):
        mock_post.side_effect = ConnectTimeout("Request has timed out.")
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(TimeoutError):
            ws.run(query="q")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_timeout_error_async(self, mock_post):
        mock_post.side_effect = ConnectTimeout("Request has timed out.")
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(TimeoutError):
            await ws.run_async(query="q")

    @patch("httpx.post")
    def test_request_exception(self, mock_post):
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(GroundRouteError):
            ws.run(query="q")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_request_exception_async(self, mock_post):
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(GroundRouteError):
            await ws.run_async(query="q")

    @patch("httpx.post")
    def test_bad_response_code(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "404 Not Found.", request=Request("POST", "https://example.com"), response=Response(404)
        )
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(GroundRouteError):
            ws.run(query="q")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_bad_response_code_async(self, mock_run):
        mock_client = AsyncMock()
        mock_response = Mock(status_code=404)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "404 Not Found.", request=Request("POST", "https://example.com"), response=Response(404)
        )
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_run.return_value = mock_client
        ws = GroundRouteWebSearch(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(GroundRouteError):
            await ws.run_async(query="q")

    @pytest.mark.skipif(
        not os.environ.get("GROUNDROUTE_API_KEY"),
        reason="Export GROUNDROUTE_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        ws = GroundRouteWebSearch(api_key=Secret.from_env_var("GROUNDROUTE_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)

    @pytest.mark.skipif(
        not os.environ.get("GROUNDROUTE_API_KEY"),
        reason="Export GROUNDROUTE_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ws = GroundRouteWebSearch(api_key=Secret.from_env_var("GROUNDROUTE_API_KEY"), top_k=3)
        result = await ws.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)
