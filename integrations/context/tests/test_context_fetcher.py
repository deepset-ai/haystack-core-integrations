# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.fetchers.context import ContextFetcher

SCRAPE_RESPONSE = {
    "success": True,
    "markdown": "# Haystack",
    "contentLength": 10,
    "url": "https://haystack.deepset.ai",
    "metadata": {"title": "Haystack", "description": "Build AI applications."},
}


class TestContextFetcher:
    def test_init_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_API_KEY", "test-key")
        fetcher = ContextFetcher()

        assert fetcher.api_key.resolve_value() == "test-key"
        assert fetcher.scrape_params is None
        assert fetcher.timeout == 60

    def test_serialization_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_API_KEY", "test-key")
        fetcher = ContextFetcher(scrape_params={"includeImages": True}, timeout=45)

        data = component_to_dict(fetcher, name="fetcher")
        restored = component_from_dict(ContextFetcher, data, name="fetcher")

        assert restored.scrape_params == {"includeImages": True}
        assert restored.timeout == 45
        assert restored.api_key.resolve_value() == "test-key"

    @patch("haystack_integrations.components.fetchers.context.context_fetcher.request_context")
    def test_run_fetches_each_url(self, request_context) -> None:
        request_context.return_value = SCRAPE_RESPONSE
        scrape_params = {"maxAgeMs": 0}
        original_params = deepcopy(scrape_params)
        fetcher = ContextFetcher(api_key=Secret.from_token("test-key"), scrape_params=scrape_params)

        result = fetcher.run(urls=["https://haystack.deepset.ai", "https://docs.haystack.deepset.ai"])

        assert len(result["documents"]) == 2
        assert all(isinstance(document, Document) for document in result["documents"])
        assert result["documents"][0].content == "# Haystack"
        assert result["documents"][0].meta == {
            "title": "Haystack",
            "description": "Build AI applications.",
            "url": "https://haystack.deepset.ai",
            "content_length": 10,
        }
        assert request_context.call_count == 2
        assert request_context.call_args_list[0].kwargs["params"] == {
            "url": "https://haystack.deepset.ai",
            "useMainContentOnly": True,
            "includeLinks": True,
            "includeImages": False,
            "maxAgeMs": 0,
        }
        assert scrape_params == original_params

    @patch("haystack_integrations.components.fetchers.context.context_fetcher.request_context")
    def test_run_runtime_params_replace_init_params(self, request_context) -> None:
        request_context.return_value = SCRAPE_RESPONSE
        fetcher = ContextFetcher(
            api_key=Secret.from_token("test-key"),
            scrape_params={"maxAgeMs": 1000},
        )

        fetcher.run(urls=["https://haystack.deepset.ai"], scrape_params={"includeImages": True})

        params = request_context.call_args.kwargs["params"]
        assert params["includeImages"] is True
        assert "maxAgeMs" not in params

    @patch("haystack_integrations.components.fetchers.context.context_fetcher.request_context_async")
    @pytest.mark.asyncio
    async def test_run_async_fetches_concurrently(self, request_context) -> None:
        request_context.side_effect = AsyncMock(return_value=SCRAPE_RESPONSE)
        fetcher = ContextFetcher(api_key=Secret.from_token("test-key"))

        result = await fetcher.run_async(urls=["https://haystack.deepset.ai", "https://docs.haystack.deepset.ai"])

        assert len(result["documents"]) == 2
        assert request_context.await_count == 2

    @pytest.mark.skipif(
        not os.environ.get("CONTEXT_API_KEY"),
        reason="Export CONTEXT_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self) -> None:
        fetcher = ContextFetcher()
        result = fetcher.run(urls=["https://haystack.deepset.ai"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content
        assert result["documents"][0].meta["url"] == "https://haystack.deepset.ai"

    @pytest.mark.skipif(
        not os.environ.get("CONTEXT_API_KEY"),
        reason="Export CONTEXT_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self) -> None:
        fetcher = ContextFetcher()
        result = await fetcher.run_async(urls=["https://haystack.deepset.ai"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content
        assert result["documents"][0].meta["url"] == "https://haystack.deepset.ai"
