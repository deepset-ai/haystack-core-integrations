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

from haystack_integrations.components.websearch.context import ContextWebSearch

SEARCH_RESPONSE = {
    "query": "Haystack",
    "results": [
        {
            "url": "https://haystack.deepset.ai",
            "title": "Haystack",
            "description": "Build production-ready AI applications.",
            "relevance": "high",
            "markdown": {"markdown": None, "code": "NOT_REQUESTED"},
        },
        {
            "url": "https://docs.haystack.deepset.ai",
            "title": "Haystack documentation",
            "description": "Haystack documentation.",
            "relevance": "medium",
            "markdown": {"markdown": "# Haystack docs", "code": "SUCCESS"},
        },
    ],
}


class TestContextWebSearch:
    def test_init_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_API_KEY", "test-key")
        websearch = ContextWebSearch()

        assert websearch.api_key.resolve_value() == "test-key"
        assert websearch.top_k == 10
        assert websearch.include_markdown is False
        assert websearch.api_url == "https://api.context.dev/v1"

    def test_init_rejects_invalid_top_k(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            ContextWebSearch(api_key=Secret.from_token("test-key"), top_k=0)

    def test_serialization_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_API_KEY", "test-key")
        websearch = ContextWebSearch(
            top_k=5,
            include_domains=["haystack.deepset.ai"],
            include_markdown=True,
            search_params={"tags": ["haystack"]},
        )

        data = component_to_dict(websearch, name="websearch")
        restored = component_from_dict(ContextWebSearch, data, name="websearch")

        assert restored.top_k == 5
        assert restored.include_domains == ["haystack.deepset.ai"]
        assert restored.include_markdown is True
        assert restored.search_params == {"tags": ["haystack"]}
        assert restored.api_key.resolve_value() == "test-key"

    @patch("haystack_integrations.components.websearch.context.context_websearch.request_context")
    def test_run_builds_request_and_returns_documents(self, request_context) -> None:
        request_context.return_value = SEARCH_RESPONSE
        search_params = {"tags": ["haystack"]}
        original_params = deepcopy(search_params)
        websearch = ContextWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=2,
            include_domains=["deepset.ai"],
            exclude_domains=["example.com"],
            freshness="last_week",
            country="us",
            include_markdown=True,
            search_params=search_params,
        )

        result = websearch.run(query="Haystack")

        request_body = request_context.call_args.kwargs["json"]
        assert request_body == {
            "query": "Haystack",
            "numResults": 10,
            "includeDomains": ["deepset.ai"],
            "excludeDomains": ["example.com"],
            "freshness": "last_week",
            "country": "us",
            "tags": ["haystack"],
            "markdownOptions": {
                "enabled": True,
                "useMainContentOnly": True,
                "includeLinks": True,
                "includeImages": False,
            },
        }
        assert search_params == original_params
        assert result["links"] == ["https://haystack.deepset.ai", "https://docs.haystack.deepset.ai"]
        assert all(isinstance(document, Document) for document in result["documents"])
        assert result["documents"][0].content == "Build production-ready AI applications."
        assert result["documents"][1].content == "# Haystack docs"
        assert result["documents"][1].meta["markdown_code"] == "SUCCESS"

    @patch("haystack_integrations.components.websearch.context.context_websearch.request_context")
    def test_run_runtime_params_replace_init_params(self, request_context) -> None:
        request_context.return_value = SEARCH_RESPONSE
        websearch = ContextWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=10,
            search_params={"tags": ["init"]},
        )

        result = websearch.run(query="Haystack", top_k=1, search_params={"tags": ["run"]})

        assert request_context.call_args.kwargs["json"]["tags"] == ["run"]
        assert len(result["documents"]) == 1

    @patch("haystack_integrations.components.websearch.context.context_websearch.request_context_async")
    @pytest.mark.asyncio
    async def test_run_async(self, request_context) -> None:
        request_context.side_effect = AsyncMock(return_value=SEARCH_RESPONSE)
        websearch = ContextWebSearch(api_key=Secret.from_token("test-key"), top_k=1)

        result = await websearch.run_async(query="Haystack")

        assert len(result["documents"]) == 1
        assert result["links"] == ["https://haystack.deepset.ai"]
        request_context.assert_awaited_once()

    def test_parse_response_skips_invalid_results(self) -> None:
        result = ContextWebSearch._parse_response({"results": [None, "invalid"]}, top_k=10)
        assert result == {"documents": [], "links": []}

    @pytest.mark.skipif(
        not os.environ.get("CONTEXT_API_KEY"),
        reason="Export CONTEXT_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self) -> None:
        websearch = ContextWebSearch(top_k=3)
        result = websearch.run(query="What is Haystack by deepset?")

        assert len(result["documents"]) == 3
        assert len(result["links"]) == 3
        assert all(document.content for document in result["documents"])

    @pytest.mark.skipif(
        not os.environ.get("CONTEXT_API_KEY"),
        reason="Export CONTEXT_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self) -> None:
        websearch = ContextWebSearch(top_k=3)
        result = await websearch.run_async(query="What is Haystack by deepset?")

        assert len(result["documents"]) == 3
        assert len(result["links"]) == 3
        assert all(document.content for document in result["documents"])
