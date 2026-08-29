# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
import requests
from haystack import ComponentError
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry

API_KEY_ENV_VAR = "CONTEXT_API_KEY"
DEFAULT_API_URL = "https://api.context.dev/v1"

try:
    _VERSION = version("context-haystack")
except PackageNotFoundError:  # pragma: no cover
    _VERSION = "0.0.0-dev"

USER_AGENT = f"context-haystack/{_VERSION} contextdev-integration/deepset-ai-haystack-core-integrations"


class ContextError(ComponentError):
    """An error occurred while calling the Context.dev API."""


def request_context(
    *,
    api_key: Secret,
    api_url: str,
    method: str,
    path: str,
    timeout: int,
    max_retries: int,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Call a Context.dev API endpoint.

    :param api_key: Context.dev API key.
    :param api_url: Base URL for the Context.dev API.
    :param method: HTTP method.
    :param path: Endpoint path relative to the API base URL.
    :param timeout: Request timeout in seconds.
    :param max_retries: Maximum number of retry attempts on transient failures.
    :param params: Optional query parameters.
    :param json: Optional JSON request body.
    :returns: Parsed JSON response.
    :raises ContextError: If the request fails or the API returns a non-object response.
    """
    try:
        response = request_with_retry(
            attempts=max_retries,
            method=method,
            url=_endpoint_url(api_url, path),
            headers=_headers(api_key),
            params=_query_params(params),
            json=json,
            timeout=timeout,
        )
    except (httpx.HTTPError, requests.RequestException) as error:
        raise ContextError(_error_message(error)) from error
    return _response_json(response)


async def request_context_async(
    *,
    api_key: Secret,
    api_url: str,
    method: str,
    path: str,
    timeout: int,
    max_retries: int,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Asynchronously call a Context.dev API endpoint.

    :param api_key: Context.dev API key.
    :param api_url: Base URL for the Context.dev API.
    :param method: HTTP method.
    :param path: Endpoint path relative to the API base URL.
    :param timeout: Request timeout in seconds.
    :param max_retries: Maximum number of retry attempts on transient failures.
    :param params: Optional query parameters.
    :param json: Optional JSON request body.
    :returns: Parsed JSON response.
    :raises ContextError: If the request fails or the API returns a non-object response.
    """
    try:
        response = await async_request_with_retry(
            attempts=max_retries,
            method=method,
            url=_endpoint_url(api_url, path),
            headers=_headers(api_key),
            params=_query_params(params),
            json=json,
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise ContextError(_error_message(error)) from error
    return _response_json(response)


def _endpoint_url(api_url: str, path: str) -> str:
    return f"{api_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(api_key: Secret) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key.resolve_value()}",
        "User-Agent": USER_AGENT,
    }


def _query_params(params: dict[str, Any] | None) -> list[tuple[str, str]] | None:
    if params is None:
        return None
    return [pair for name, value in params.items() for pair in _query_value(name, value)]


def _query_value(name: str, value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [pair for key, item in value.items() for pair in _query_value(f"{name}[{key}]", item)]
    if isinstance(value, (list, tuple)):
        return [pair for item in value for pair in _query_value(name, item)]
    if isinstance(value, bool):
        return [(name, "true" if value else "false")]
    return [(name, str(value))]


def _response_json(response: httpx.Response | requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        msg = "Context.dev API returned an invalid JSON response."
        raise ContextError(msg) from error
    if not isinstance(payload, dict):
        msg = "Context.dev API returned an invalid JSON response."
        raise ContextError(msg)
    return payload


def _error_message(error: httpx.HTTPError | requests.RequestException) -> str:
    message = f"An error occurred while calling the Context.dev API. Error: {error}"
    response = getattr(error, "response", None)
    if response is not None:
        return f"{message}, Response: {response.text}"
    return message
