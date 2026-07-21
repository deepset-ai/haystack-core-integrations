# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.websearch.linkup import LinkupWebSearch


def _search_results():
    """Mimic a Linkup `LinkupSearchResults` object with text and image results."""
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                type="text",
                name="Example Title",
                url="https://example.com",
                content="Example content",
            ),
            SimpleNamespace(
                type="image",
                name="Example Image",
                url="https://example.com/image.png",
            ),
        ]
    )


class TestLinkupWebSearch:
    @pytest.fixture
    def search_response(self):
        return _search_results()

    @pytest.fixture
    def mock_client(self, search_response):
        client = MagicMock()
        client.search.return_value = search_response
        client.async_search = AsyncMock(return_value=search_response)
        return client

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("LINKUP_API_KEY", "test-key")
        ws = LinkupWebSearch()
        assert ws.top_k == 10
        assert ws.depth == "standard"
        assert ws.search_params is None
        assert ws.api_key.resolve_value() == "test-key"

    def test_init_with_params(self):
        ws = LinkupWebSearch(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            depth="deep",
            search_params={"include_images": True},
        )
        assert ws.top_k == 5
        assert ws.depth == "deep"
        assert ws.search_params == {"include_images": True}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("LINKUP_API_KEY", "test-key")
        ws = LinkupWebSearch(top_k=5, depth="deep", search_params={"include_images": True})
        data = component_to_dict(ws, "LinkupWebSearch")
        assert data["type"] == "haystack_integrations.components.websearch.linkup.linkup_websearch.LinkupWebSearch"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["depth"] == "deep"
        assert data["init_parameters"]["search_params"] == {"include_images": True}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("LINKUP_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.websearch.linkup.linkup_websearch.LinkupWebSearch",
            "init_parameters": {
                "top_k": 3,
                "depth": "standard",
                "search_params": {"include_domains": ["example.com"]},
                "api_key": {"env_vars": ["LINKUP_API_KEY"], "strict": True, "type": "env_var"},
            },
        }
        ws = component_from_dict(LinkupWebSearch, data, "LinkupWebSearch")
        assert ws.top_k == 3
        assert ws.depth == "standard"
        assert ws.search_params == {"include_domains": ["example.com"]}

    def test_run_returns_documents_and_links(self, mock_client):
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._client = mock_client

        result = ws.run(query="test query")

        assert len(result["documents"]) == 2
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "Example content"
        assert result["documents"][0].meta["url"] == "https://example.com"
        assert result["documents"][0].meta["title"] == "Example Title"
        # Image results have no content but still contribute a link.
        assert result["documents"][1].content == ""
        assert result["links"] == ["https://example.com", "https://example.com/image.png"]
        mock_client.search.assert_called_once_with(
            query="test query", depth="standard", output_type="searchResults", max_results=10
        )

    def test_run_overrides_params_at_runtime(self, mock_client):
        ws = LinkupWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"include_images": False},
        )
        ws._client = mock_client

        ws.run(query="test", search_params={"include_images": True, "max_results": 3})

        mock_client.search.assert_called_once_with(
            query="test", depth="standard", output_type="searchResults", include_images=True, max_results=3
        )

    @pytest.mark.asyncio
    async def test_run_async(self, mock_client):
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"), top_k=10)
        ws._client = mock_client

        result = await ws.run_async(query="test query")

        assert len(result["documents"]) == 2
        assert result["links"] == ["https://example.com", "https://example.com/image.png"]
        mock_client.async_search.assert_awaited_once_with(
            query="test query", depth="standard", output_type="searchResults", max_results=10
        )

    def test_run_raises_on_error(self, mock_client):
        mock_client.search.side_effect = Exception("API error")
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
        ws._client = mock_client

        with pytest.raises(Exception, match="API error"):
            ws.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async_raises_on_error(self, mock_client):
        mock_client.async_search = AsyncMock(side_effect=Exception("API error"))
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
        ws._client = mock_client

        with pytest.raises(Exception, match="API error"):
            await ws.run_async(query="test")

    def test_warm_up_initializes_client(self):
        with patch("haystack_integrations.components.websearch.linkup.linkup_websearch.LinkupClient") as mock_cls:
            ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
            assert ws._client is None
            ws.warm_up()
            assert ws._client is not None
            mock_cls.assert_called_once_with(api_key="test-key")

    def test_run_triggers_warm_up(self, search_response):
        with patch("haystack_integrations.components.websearch.linkup.linkup_websearch.LinkupClient") as mock_cls:
            mock_cls.return_value.search.return_value = search_response
            ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
            ws.run(query="test")
            mock_cls.assert_called_once_with(api_key="test-key")

    def test_run_empty_results(self, mock_client):
        mock_client.search.return_value = SimpleNamespace(results=[])
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
        ws._client = mock_client
        result = ws.run(query="obscure query")
        assert result["documents"] == []
        assert result["links"] == []

    def test_run_raises_runtime_error_when_warm_up_fails_to_initialize_client(self, monkeypatch):
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(ws, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="LinkupWebSearch client failed to initialize"):
            ws.run(query="test")

    @pytest.mark.asyncio
    async def test_run_async_raises_runtime_error_when_warm_up_fails_to_initialize_client(self, monkeypatch):
        ws = LinkupWebSearch(api_key=Secret.from_token("test-key"))
        monkeypatch.setattr(ws, "warm_up", lambda: None)
        with pytest.raises(RuntimeError, match="LinkupWebSearch client failed to initialize"):
            await ws.run_async(query="test")

    @pytest.mark.skipif(
        not os.environ.get("LINKUP_API_KEY"),
        reason="Export LINKUP_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        ws = LinkupWebSearch(api_key=Secret.from_env_var("LINKUP_API_KEY"), top_k=3)
        result = ws.run(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert isinstance(result["documents"][0], Document)

    @pytest.mark.skipif(
        not os.environ.get("LINKUP_API_KEY"),
        reason="Export LINKUP_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ws = LinkupWebSearch(api_key=Secret.from_env_var("LINKUP_API_KEY"), top_k=3)
        result = await ws.run_async(query="What is Haystack by deepset?")
        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
