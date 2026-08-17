# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.youcom import YouComError, YouComWebSearch
from haystack_integrations.components.websearch.youcom.youcom_websearch import (
    API_KEY_ENV_VARS,
    USER_AGENT,
    YOUCOM_KEYED_SEARCH_URL,
    YOUCOM_KEYLESS_SEARCH_URL,
)

SAMPLE_RESPONSE = {
    "results": {
        "web": [
            {
                "url": "https://haystack.deepset.ai",
                "title": "Haystack | Haystack",
                "description": "Open-source AI framework",
                "snippets": ["Haystack is an open-source framework", "for building production-ready AI pipelines"],
                "page_age": "2026-07-01T00:00:00",
            },
        ],
        "news": [
            {
                "url": "https://example.com/news",
                "title": "AI news item",
                "description": "A news description",
                "page_age": "2026-07-20T00:00:00",
            },
        ],
    },
    "metadata": {"query": "test", "latency": 0.1},
}


def mock_response(json_body, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    return response


def _keyed_env():
    """True when any accepted API key env var holds a value, so the keyed live tests can run."""
    return any(os.environ.get(env_var) for env_var in API_KEY_ENV_VARS)


def http_status_error(status_code, message="Client Error"):
    """Build the `httpx.HTTPStatusError` that `request_with_retry` raises for an error response."""
    request = httpx.Request("GET", YOUCOM_KEYLESS_SEARCH_URL)
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"{status_code} {message}", request=request, response=response)


class TestYouComWebSearch:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        for env_var in API_KEY_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)

    def test_init_defaults(self):
        component = YouComWebSearch()
        assert component.api_key == Secret.from_env_var(API_KEY_ENV_VARS, strict=False)
        assert component.keyless_fallback is True
        assert component.top_k == 10
        assert component.freshness is None
        assert component.country is None
        assert component.search_lang is None
        assert component.safesearch is None
        assert component.extra_params is None
        assert component.timeout == 10
        assert component.max_retries == 3

    def test_init_without_env_var_does_not_raise(self):
        component = YouComWebSearch()
        assert component.api_key.resolve_value() is None

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyless_when_no_api_key(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYLESS_SEARCH_URL
        assert "X-API-Key" not in kwargs["headers"]

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyed_when_env_var_set(self, mock_request, monkeypatch):
        monkeypatch.setenv("YOUDOTCOM_API_KEY", "env-key")
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYED_SEARCH_URL
        assert kwargs["headers"]["X-API-Key"] == "env-key"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyed_when_legacy_env_var_set(self, mock_request, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "legacy-key")
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYED_SEARCH_URL
        assert kwargs["headers"]["X-API-Key"] == "legacy-key"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_primary_env_var_wins_over_legacy(self, mock_request, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "legacy-key")
        monkeypatch.setenv("YOUDOTCOM_API_KEY", "primary-key")
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        assert mock_request.call_args.kwargs["headers"]["X-API-Key"] == "primary-key"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_keyless_fallback_disabled_raises_without_key(self, mock_request):
        component = YouComWebSearch(keyless_fallback=False)

        with pytest.raises(YouComError, match="keyless_fallback"):
            component.run(query="test")
        mock_request.assert_not_called()

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_keyless_fallback_disabled_still_searches_with_key(self, mock_request, monkeypatch):
        monkeypatch.setenv("YOUDOTCOM_API_KEY", "env-key")
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(keyless_fallback=False)
        component.run(query="test")

        assert mock_request.call_args.kwargs["url"] == YOUCOM_KEYED_SEARCH_URL

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.async_request_with_retry")
    async def test_keyless_fallback_disabled_raises_without_key_async(self, mock_request):
        component = YouComWebSearch(keyless_fallback=False)

        with pytest.raises(YouComError, match="keyless_fallback"):
            await component.run_async(query="test")
        mock_request.assert_not_called()

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyed_with_explicit_token(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(api_key=Secret.from_token("explicit-key"))
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYED_SEARCH_URL
        assert kwargs["headers"]["X-API-Key"] == "explicit-key"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_sends_user_agent(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        headers = mock_request.call_args.kwargs["headers"]
        assert headers["User-Agent"] == USER_AGENT
        assert USER_AGENT.startswith("youcom-haystack/")
        assert USER_AGENT.endswith(" youdotcom-integration/deepset-ai-haystack-core-integrations")

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_builds_params(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(
            top_k=5,
            freshness="week",
            country="US",
            search_lang="EN",
            safesearch="moderate",
            extra_params={"include_domains": "nytimes.com,bbc.com"},
        )
        component.run(query="climate news")

        params = mock_request.call_args.kwargs["params"]
        assert params == {
            "query": "climate news",
            "count": 5,
            "freshness": "week",
            "country": "US",
            "language": "EN",
            "safesearch": "moderate",
            "include_domains": "nytimes.com,bbc.com",
        }

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_top_k_override(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(top_k=10)
        component.run(query="q", top_k=3)

        assert mock_request.call_args.kwargs["params"]["count"] == 3

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_parses_documents_and_links(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        result = component.run(query="test")

        documents = result["documents"]
        links = result["links"]
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        web_doc = documents[0]
        assert web_doc.content == "Haystack is an open-source framework\nfor building production-ready AI pipelines"
        assert web_doc.meta == {
            "title": "Haystack | Haystack",
            "url": "https://haystack.deepset.ai",
            "source": "web",
            "page_age": "2026-07-01T00:00:00",
        }

        news_doc = documents[1]
        assert news_doc.content == "A news description"
        assert news_doc.meta["source"] == "news"

        assert links == ["https://haystack.deepset.ai", "https://example.com/news"]

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_empty_results(self, mock_request):
        mock_request.return_value = mock_response({"results": {"web": [], "news": []}})

        component = YouComWebSearch()
        result = component.run(query="test")

        assert result == {"documents": [], "links": []}

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_missing_sections(self, mock_request):
        mock_request.return_value = mock_response({"metadata": {}})

        component = YouComWebSearch()
        result = component.run(query="test")

        assert result == {"documents": [], "links": []}

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_null_sections(self, mock_request):
        mock_request.return_value = mock_response({"results": {"web": None, "news": None}})

        component = YouComWebSearch()
        result = component.run(query="test")

        assert result == {"documents": [], "links": []}

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_keyless_rate_limit_error_includes_upgrade_hint(self, mock_request):
        mock_request.side_effect = http_status_error(402)

        component = YouComWebSearch()
        with pytest.raises(YouComError, match="YOUDOTCOM_API_KEY"):
            component.run(query="test")

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_keyed_error_is_not_rewritten(self, mock_request):
        mock_request.side_effect = http_status_error(401, "Unauthorized")

        component = YouComWebSearch(api_key=Secret.from_token("bad-key"))
        with pytest.raises(YouComError, match="401") as excinfo:
            component.run(query="test")
        assert "YOUDOTCOM_API_KEY" not in str(excinfo.value)

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_network_error_is_wrapped(self, mock_request):
        """Transport-level failures have no response, so they are reported without one."""
        mock_request.side_effect = httpx.ConnectError("connection refused")

        component = YouComWebSearch()
        with pytest.raises(YouComError, match="connection refused") as excinfo:
            component.run(query="test")
        assert "Response:" not in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, httpx.ConnectError)

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.async_request_with_retry")
    async def test_run_async(self, mock_request):
        mock_request.side_effect = AsyncMock(return_value=mock_response(SAMPLE_RESPONSE))

        component = YouComWebSearch()
        result = await component.run_async(query="test")

        assert len(result["documents"]) == 2
        assert mock_request.call_args.kwargs["url"] == YOUCOM_KEYLESS_SEARCH_URL

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.async_request_with_retry")
    async def test_run_async_keyless_rate_limit_error_includes_upgrade_hint(self, mock_request):
        mock_request.side_effect = AsyncMock(side_effect=http_status_error(429, "Too Many Requests"))

        component = YouComWebSearch()
        with pytest.raises(YouComError, match="YOUDOTCOM_API_KEY"):
            await component.run_async(query="test")

    def test_serialization_roundtrip_without_env_var(self):
        component = YouComWebSearch(top_k=7, freshness="month", keyless_fallback=False)
        data = component_to_dict(component, name="websearch")
        restored = component_from_dict(YouComWebSearch, data, name="websearch")

        assert data["init_parameters"]["keyless_fallback"] is False
        assert restored.top_k == 7
        assert restored.freshness == "month"
        assert restored.keyless_fallback is False
        assert restored.api_key == Secret.from_env_var(API_KEY_ENV_VARS, strict=False)

    def test_serialization_roundtrip_with_env_var(self, monkeypatch):
        monkeypatch.setenv("YOUDOTCOM_API_KEY", "env-key")
        component = YouComWebSearch()
        data = component_to_dict(component, name="websearch")
        restored = component_from_dict(YouComWebSearch, data, name="websearch")

        assert restored.api_key.resolve_value() == "env-key"

    def test_serialization_does_not_leak_token_secret(self):
        component = YouComWebSearch(api_key=Secret.from_token("super-secret"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component_to_dict(component, name="websearch")


@pytest.mark.integration
class TestYouComWebSearchIntegration:
    def test_keyless_live_search(self, monkeypatch):
        """The keyless free tier is rate limited per IP, so a shared CI runner may legitimately hit the limit."""
        for env_var in API_KEY_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        component = YouComWebSearch(top_k=2)
        try:
            result = component.run(query="What is Haystack by deepset?")
        except YouComError as error:
            cause = error.__cause__
            status = cause.response.status_code if isinstance(cause, httpx.HTTPStatusError) else None
            if status in (402, 429):
                pytest.skip(f"keyless free-tier rate limit hit: {status}")
            raise

        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert result["documents"][0].content

    @pytest.mark.skipif(not _keyed_env(), reason="no You.com API key in the environment")
    def test_keyed_live_search(self):
        component = YouComWebSearch(top_k=2)
        result = component.run(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0

    @pytest.mark.skipif(not _keyed_env(), reason="no You.com API key in the environment")
    async def test_keyed_live_search_async(self):
        component = YouComWebSearch(top_k=2)
        result = await component.run_async(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
