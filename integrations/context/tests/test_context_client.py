# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.context._client import USER_AGENT, ContextError, request_context, request_context_async


def test_request_context_builds_authenticated_request() -> None:
    response = MagicMock()
    response.json.return_value = {"results": []}

    with patch("haystack_integrations.context._client.request_with_retry", return_value=response) as request:
        result = request_context(
            api_key=Secret.from_token("test-key"),
            api_url="https://example.test/v1/",
            method="POST",
            path="/web/search",
            json={"query": "Haystack"},
            timeout=30,
            max_retries=2,
        )

    assert result == {"results": []}
    assert request.call_args.kwargs == {
        "attempts": 2,
        "method": "POST",
        "url": "https://example.test/v1/web/search",
        "headers": {
            "Accept": "application/json",
            "Authorization": "Bearer test-key",
            "User-Agent": USER_AGENT,
        },
        "params": None,
        "json": {"query": "Haystack"},
        "timeout": 30,
    }


@pytest.mark.asyncio
async def test_request_context_async_builds_authenticated_request() -> None:
    response = MagicMock()
    response.json.return_value = {"success": True}

    with patch(
        "haystack_integrations.context._client.async_request_with_retry",
        new=AsyncMock(return_value=response),
    ) as request:
        result = await request_context_async(
            api_key=Secret.from_token("test-key"),
            api_url="https://example.test/v1",
            method="GET",
            path="web/scrape/markdown",
            params={"url": "https://example.com"},
            timeout=60,
            max_retries=3,
        )

    assert result == {"success": True}
    request.assert_awaited_once()
    assert request.call_args.kwargs["url"] == "https://example.test/v1/web/scrape/markdown"


def test_request_context_wraps_http_errors() -> None:
    request = httpx.Request("POST", "https://example.test/v1/web/search")
    response = httpx.Response(401, text="invalid api key", request=request)
    error = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

    with (
        patch("haystack_integrations.context._client.request_with_retry", side_effect=error),
        pytest.raises(ContextError, match="invalid api key") as exc_info,
    ):
        request_context(
            api_key=Secret.from_token("bad-key"),
            api_url="https://example.test/v1",
            method="POST",
            path="/web/search",
            timeout=30,
            max_retries=1,
        )

    assert exc_info.value.__cause__ is error


def test_request_context_wraps_requests_http_errors() -> None:
    response = requests.Response()
    response.status_code = 400
    response._content = b"invalid query"
    error = requests.HTTPError("400 Bad Request", response=response)

    with (
        patch("haystack_integrations.context._client.request_with_retry", side_effect=error),
        pytest.raises(ContextError, match="invalid query") as exc_info,
    ):
        request_context(
            api_key=Secret.from_token("test-key"),
            api_url="https://example.test/v1",
            method="GET",
            path="/web/scrape/markdown",
            timeout=30,
            max_retries=1,
        )

    assert exc_info.value.__cause__ is error


def test_request_context_serializes_nested_query_parameters() -> None:
    response = MagicMock()
    response.json.return_value = {"success": True}

    with patch("haystack_integrations.context._client.request_with_retry", return_value=response) as request:
        request_context(
            api_key=Secret.from_token("test-key"),
            api_url="https://example.test/v1",
            method="GET",
            path="/web/scrape/markdown",
            params={
                "includeLinks": True,
                "includeSelectors": ["main", "article"],
                "pdf": {"start": 2, "ocr": False},
                "unset": None,
            },
            timeout=30,
            max_retries=1,
        )

    assert request.call_args.kwargs["params"] == [
        ("includeLinks", "true"),
        ("includeSelectors", "main"),
        ("includeSelectors", "article"),
        ("pdf[start]", "2"),
        ("pdf[ocr]", "false"),
    ]


def test_request_context_rejects_non_object_json() -> None:
    response = MagicMock()
    response.json.return_value = []

    with (
        patch("haystack_integrations.context._client.request_with_retry", return_value=response),
        pytest.raises(ContextError, match="invalid JSON response"),
    ):
        request_context(
            api_key=Secret.from_token("test-key"),
            api_url="https://example.test/v1",
            method="GET",
            path="/web/scrape/markdown",
            timeout=30,
            max_retries=1,
        )


def test_request_context_wraps_invalid_json() -> None:
    response = MagicMock()
    response.json.side_effect = ValueError("invalid json")

    with (
        patch("haystack_integrations.context._client.request_with_retry", return_value=response),
        pytest.raises(ContextError, match="invalid JSON response") as exc_info,
    ):
        request_context(
            api_key=Secret.from_token("test-key"),
            api_url="https://example.test/v1",
            method="GET",
            path="/web/scrape/markdown",
            timeout=30,
            max_retries=1,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
