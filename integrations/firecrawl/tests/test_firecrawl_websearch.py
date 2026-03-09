# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.firecrawl import FirecrawlWebSearch


class TestFirecrawlWebSearch:
    @pytest.fixture
    def search_result_web(self) -> MagicMock:
        """A SearchResultWeb-like object (no markdown, just search metadata)."""
        result = MagicMock(spec=["url", "title", "description"])
        result.url = "https://example.com"
        result.title = "Example Title"
        result.description = "Example description snippet"
        return result

    @pytest.fixture
    def search_result_document(self) -> MagicMock:
        """A Document-like object (with markdown, from scrapeOptions)."""
        result = MagicMock(spec=["markdown", "metadata_dict", "url"])
        result.markdown = "# Full page content\nSome markdown text."
        result.metadata_dict = {"url": "https://example.com/page", "title": "Page Title"}
        result.url = "https://example.com/page"
        return result

    @pytest.fixture
    def search_response(self, search_result_web) -> MagicMock:
        """Standard search response with web results."""
        response = MagicMock()
        response.web = [search_result_web]
        return response

    @pytest.fixture
    def mock_client(self, search_response) -> MagicMock:
        client = MagicMock()
        client.search.return_value = search_response
        return client

    @pytest.fixture
    def mock_async_client(self, search_response) -> MagicMock:
        client = MagicMock()
        client.search = AsyncMock(return_value=search_response)
        return client

    def test_init_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        ws = FirecrawlWebSearch()
        assert ws.top_k == 10
        assert ws._search_params == {}
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_params(self) -> None:
        ws = FirecrawlWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            search_params={"tbs": "qdr:d", "location": "US"},
        )
        assert ws.top_k == 5
        assert ws._search_params == {"tbs": "qdr:d", "location": "US"}
        assert ws.api_key.resolve_value() == "custom-key"

    def test_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        ws = FirecrawlWebSearch(top_k=5, search_params={"tbs": "qdr:d"})
        data = component_to_dict(ws, "FirecrawlWebSearch")
        assert (
            data["type"]
            == "haystack_integrations.components.websearch.firecrawl.firecrawl_websearch.FirecrawlWebSearch"
        )
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["search_params"] == {"tbs": "qdr:d"}

    def test_from_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        data = {
            "type": ("haystack_integrations.components.websearch.firecrawl.firecrawl_websearch.FirecrawlWebSearch"),
            "init_parameters": {
                "top_k": 3,
                "search_params": {"location": "UK"},
                "api_key": {"env_vars": ["FIRECRAWL_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        ws = component_from_dict(FirecrawlWebSearch, data, "FirecrawlWebSearch")
        assert ws.top_k == 3
        assert ws.search_params == {"location": "UK"}
        assert ws.api_key.resolve_value() == "test-key"

    def test_run_returns_documents_and_links(self, mock_client) -> None:
        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._firecrawl_client = mock_client

        result = ws.run(query="test query")

        assert "documents" in result
        assert "links" in result
        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example description snippet"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        assert result["links"] == ["https://example.com"]
        mock_client.search.assert_called_once_with(query="test query", limit=10)

    def test_run_with_scraped_documents(self, search_result_document, mock_client) -> None:
        """When scrapeOptions are used, results contain markdown content."""
        search_response = MagicMock()
        search_response.web = [search_result_document]
        mock_client.search.return_value = search_response

        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._firecrawl_client = mock_client

        result = ws.run(query="test query")

        assert result["documents"][0].content == "# Full page content\nSome markdown text."
        assert result["documents"][0].meta["url"] == "https://example.com/page"
        assert result["documents"][0].meta["title"] == "Page Title"

    def test_run_overrides_init_params_with_runtime_params(self, mock_client) -> None:
        ws = FirecrawlWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"location": "US"},
        )
        ws._firecrawl_client = mock_client

        ws.run(query="test", search_params={"location": "UK", "limit": 5})

        mock_client.search.assert_called_once_with(
            query="test",
            location="UK",
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_client) -> None:
        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._async_firecrawl_client = mock_async_client

        result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["links"] == ["https://example.com"]
        mock_async_client.search.assert_awaited_once()

    def test_run_returns_empty_on_error(self, mock_client) -> None:
        mock_client.search.side_effect = Exception("API error")

        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
        ws._firecrawl_client = mock_client

        result = ws.run(query="test")
        assert result["documents"] == []
        assert result["links"] == []

    @pytest.mark.asyncio
    async def test_run_async_returns_empty_on_error(self, mock_async_client) -> None:
        mock_async_client.search = AsyncMock(side_effect=Exception("API error"))

        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
        ws._async_firecrawl_client = mock_async_client

        result = await ws.run_async(query="test")
        assert result["documents"] == []
        assert result["links"] == []

    def test_run_calls_warm_up(self, search_response) -> None:
        with (
            patch(
                "haystack_integrations.components.websearch.firecrawl.firecrawl_websearch.Firecrawl"
            ) as mock_firecrawl_client,
            patch("haystack_integrations.components.websearch.firecrawl.firecrawl_websearch.AsyncFirecrawl"),
        ):
            mock_firecrawl_client.return_value.search.return_value = search_response

            ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
            ws.run(query="test")

            assert ws._firecrawl_client is mock_firecrawl_client.return_value
            mock_firecrawl_client.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_run_async_calls_warm_up(self, search_response) -> None:
        with (
            patch("haystack_integrations.components.websearch.firecrawl.firecrawl_websearch.Firecrawl"),
            patch(
                "haystack_integrations.components.websearch.firecrawl.firecrawl_websearch.AsyncFirecrawl"
            ) as mock_async_firecrawl_client,
        ):
            mock_async_firecrawl_client.return_value.search = AsyncMock(return_value=search_response)

            ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
            await ws.run_async(query="test")

            assert ws._async_firecrawl_client is mock_async_firecrawl_client.return_value
            mock_async_firecrawl_client.assert_called_once_with(api_key="test-key")

    def test_warm_up_initializes_clients(self) -> None:
        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
        assert ws._firecrawl_client is None
        assert ws._async_firecrawl_client is None

        ws.warm_up()

        assert ws._firecrawl_client is not None
        assert ws._async_firecrawl_client is not None

    def test_run_empty_web_results(self, mock_client) -> None:
        empty_response = MagicMock()
        empty_response.web = []
        mock_client.search.return_value = empty_response

        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
        ws._firecrawl_client = mock_client
        result = ws.run(query="obscure query")
        assert result["documents"] == []
        assert result["links"] == []

    def test_run_none_web_results(self, mock_client) -> None:
        none_response = MagicMock()
        none_response.web = None
        mock_client.search.return_value = none_response

        ws = FirecrawlWebSearch(api_key=Secret.from_token("test-key"))
        ws._firecrawl_client = mock_client
        result = ws.run(query="obscure query")
        assert result["documents"] == []
        assert result["links"] == []

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"),
        reason="Export FIRECRAWL_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self) -> None:
        ws = FirecrawlWebSearch(
            api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
            top_k=3,
        )
        result = ws.run(query="What is Haystack by deepset?")

        assert "documents" in result
        assert "links" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"),
        reason="Export FIRECRAWL_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self) -> None:
        ws = FirecrawlWebSearch(
            api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
            top_k=3,
        )
        result = await ws.run_async(query="What is Haystack by deepset?")

        assert "documents" in result
        assert "links" in result
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
