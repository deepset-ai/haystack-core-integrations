# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.keirolabs import KeirolabsError, KeirolabsWebSearch
from haystack_integrations.components.websearch.keirolabs.keirolabs_websearch import KEIROLABS_SEARCH_URL

MODULE = "haystack_integrations.components.websearch.keirolabs.keirolabs_websearch"

SAMPLE_RESPONSE = {
    "query": "haystack",
    "type": "lite",
    "count": 3,
    "results": [
        {
            "title": "Haystack | Haystack",
            "url": "https://haystack.deepset.ai",
            "snippet": "Open-source AI framework",
            "domain": "haystack.deepset.ai",
            "position": 1,
            "score": None,
        },
        {"title": "No snippet", "url": "https://example.org/", "snippet": None},
        {"title": "No url", "snippet": "dropped"},
    ],
}


def mock_response(json_body):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json_body
    return response


class TestKeirolabsWebSearch:
    @pytest.fixture(autouse=True)
    def api_key(self, monkeypatch):
        monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")

    def test_init_defaults(self):
        component = KeirolabsWebSearch()
        assert component.api_key == Secret.from_env_var("KEIROLABS_API_KEY")
        assert component.top_k == 10
        assert component.allowed_domains is None

    def test_init_fails_without_api_key(self, monkeypatch):
        monkeypatch.delenv("KEIROLABS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="KEIROLABS_API_KEY"):
            KeirolabsWebSearch().run(query="test")

    def test_serialization_roundtrip(self):
        component = KeirolabsWebSearch(top_k=3, allowed_domains=["deepset.ai"], timeout=5, max_retries=1)
        data = component_to_dict(component, "websearch")
        assert data["init_parameters"]["api_key"] == {
            "type": "env_var",
            "env_vars": ["KEIROLABS_API_KEY"],
            "strict": True,
        }
        restored = component_from_dict(KeirolabsWebSearch, data, "websearch")
        assert restored.top_k == 3
        assert restored.allowed_domains == ["deepset.ai"]
        assert restored.timeout == 5
        assert restored.max_retries == 1
        assert restored.api_key.resolve_value() == "test-key"

    @patch(f"{MODULE}.request_with_retry")
    def test_run_sends_request_and_parses_results(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        result = KeirolabsWebSearch(top_k=5, allowed_domains=["deepset.ai", "example.org"]).run(query="haystack")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == KEIROLABS_SEARCH_URL
        assert kwargs["json"] == {"query": "(haystack) (site:deepset.ai OR site:example.org)", "maxResults": 5}
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert result["documents"] == [
            Document(
                content="Open-source AI framework",
                meta={
                    "title": "Haystack | Haystack",
                    "url": "https://haystack.deepset.ai",
                    "domain": "haystack.deepset.ai",
                },
            ),
            Document(content="", meta={"title": "No snippet", "url": "https://example.org/"}),
        ]
        assert result["links"] == ["https://haystack.deepset.ai", "https://example.org/"]

    @patch(f"{MODULE}.request_with_retry")
    def test_run_top_k_override_caps_results(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        result = KeirolabsWebSearch(top_k=10).run(query="haystack", top_k=1)

        assert mock_request.call_args.kwargs["json"] == {"query": "haystack", "maxResults": 1}
        assert result["links"] == ["https://haystack.deepset.ai"]

    @patch(f"{MODULE}.request_with_retry")
    def test_run_raises_keirolabs_error_on_http_error(self, mock_request):
        request = httpx.Request("POST", KEIROLABS_SEARCH_URL)
        response = httpx.Response(401, request=request, text='{"error":"Invalid API key"}')
        mock_request.side_effect = httpx.HTTPStatusError("401", request=request, response=response)

        with pytest.raises(KeirolabsError, match="Invalid API key"):
            KeirolabsWebSearch().run(query="haystack")

    @patch(f"{MODULE}.async_request_with_retry")
    async def test_run_async(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        result = await KeirolabsWebSearch().run_async(query="haystack")

        assert mock_request.call_args.kwargs["json"] == {"query": "haystack", "maxResults": 10}
        assert result["links"] == ["https://haystack.deepset.ai", "https://example.org/"]


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("KEIROLABS_API_KEY"), reason="KEIROLABS_API_KEY is not set")
class TestKeirolabsWebSearchLive:
    def test_run(self):
        result = KeirolabsWebSearch(top_k=3).run(query="What is Haystack by deepset?")
        assert 0 < len(result["documents"]) <= 3
        assert all(link.startswith("http") for link in result["links"])

    async def test_run_async(self):
        result = await KeirolabsWebSearch(top_k=3).run_async(query="What is Haystack by deepset?")
        assert 0 < len(result["documents"]) <= 3
