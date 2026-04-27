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

from haystack_integrations.components.websearch.brave import BraveWebSearch

MOCK_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Example Title",
                "url": "https://example.com",
                "description": "Example description",
            }
        ]
    }
}


class TestBraveWebSearch:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch()
        assert ws.top_k == 10
        assert ws.country is None
        assert ws.search_lang is None
        assert ws.extra_params is None
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_params(self):
        ws = BraveWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            country="US",
            search_lang="en",
            extra_params={"safesearch": "moderate"},
        )
        assert ws.top_k == 5
        assert ws.country == "US"
        assert ws.search_lang == "en"
        assert ws.extra_params == {"safesearch": "moderate"}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(top_k=5, country="US")
        data = component_to_dict(ws, "BraveWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.brave.brave_websearch.BraveWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["country"] == "US"

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.websearch.brave.brave_websearch.BraveWebSearch",
            "init_parameters": {
                "top_k": 3,
                "country": "DE",
                "search_lang": "de",
                "extra_params": None,
                "api_key": {"env_vars": ["BRAVE_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        ws = component_from_dict(BraveWebSearch, data, "BraveWebSearch")
        assert ws.top_k == 3
        assert ws.country == "DE"
        assert ws.search_lang == "de"

    def test_run_returns_documents_and_links(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"), top_k=5)

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_RESPONSE

        with patch("haystack_integrations.components.websearch.brave.brave_websearch.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
            result = ws.run(query="test query")

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example description"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["links"] == ["https://example.com"]

    def test_run_passes_correct_params(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"), top_k=5, country="US", search_lang="en")

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}

        with patch("haystack_integrations.components.websearch.brave.brave_websearch.httpx.Client") as mock_client_cls:
            mock_get = mock_client_cls.return_value.__enter__.return_value.get
            mock_get.return_value = mock_response
            ws.run(query="test query")

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs["params"]
        assert params["q"] == "test query"
        assert params["count"] == 5
        assert params["country"] == "US"
        assert params["search_lang"] == "en"

    def test_run_top_k_override(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"), top_k=10)

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}

        with patch("haystack_integrations.components.websearch.brave.brave_websearch.httpx.Client") as mock_client_cls:
            mock_get = mock_client_cls.return_value.__enter__.return_value.get
            mock_get.return_value = mock_response
            ws.run(query="test", top_k=3)

        params = mock_get.call_args.kwargs["params"]
        assert params["count"] == 3

    @pytest.mark.asyncio
    async def test_run_async_returns_documents_and_links(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"), top_k=5)

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_RESPONSE

        with patch(
            "haystack_integrations.components.websearch.brave.brave_websearch.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client_cls.return_value)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value.get = AsyncMock(return_value=mock_response)
            result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 1
        assert result["links"] == ["https://example.com"]

    def test_run_empty_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}

        with patch("haystack_integrations.components.websearch.brave.brave_websearch.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
            result = ws.run(query="very obscure query")

        assert result["documents"] == []
        assert result["links"] == []

    def test_run_missing_web_key(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {}

        with patch("haystack_integrations.components.websearch.brave.brave_websearch.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response
            result = ws.run(query="test")

        assert result["documents"] == []
        assert result["links"] == []

    def test_run_raises_on_http_error(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        ws = BraveWebSearch(api_key=Secret.from_token("test-key"))

        with patch("haystack_integrations.components.websearch.brave.brave_websearch.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value.raise_for_status.side_effect = (
                httpx.HTTPStatusError("403 Forbidden", request=MagicMock(), response=MagicMock())
            )
            with pytest.raises(httpx.HTTPStatusError):
                ws.run(query="test")

    @pytest.mark.skipif(
        not os.environ.get("BRAVE_API_KEY"),
        reason="Export BRAVE_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        ws = BraveWebSearch(api_key=Secret.from_env_var("BRAVE_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)

    @pytest.mark.skipif(
        not os.environ.get("BRAVE_API_KEY"),
        reason="Export BRAVE_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ws = BraveWebSearch(api_key=Secret.from_env_var("BRAVE_API_KEY"), top_k=3)
        result = await ws.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
