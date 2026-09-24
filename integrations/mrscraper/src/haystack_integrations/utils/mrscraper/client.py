# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal
from urllib.parse import quote

import httpx
from haystack import ComponentError

PRIMARY_ORIGIN = "https://api.app.mrscraper.com"
SERP_ORIGIN = "https://sync.scraper.mrscraper.com"
RENDERED_ORIGIN = "https://api.mrscraper.com"
_MAX_ERROR_DETAIL = 500


class MrScraperError(ComponentError):
    """An error returned while communicating with the MrScraper API."""


class MrScraperClient:
    """Stateless synchronous and asynchronous client for the fixed MrScraper API origins."""

    def __init__(self, *, api_key: str, connect_timeout: float, read_timeout: float) -> None:
        if not api_key:
            msg = "The MrScraper API token resolved to an empty value."
            raise MrScraperError(msg)
        self._api_key = api_key
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )

    @property
    def _json_headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "Content-Type": "application/json"}

    def primary(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request to the primary API."""
        headers = {**self._json_headers, "x-api-token": self._api_key}
        return self._request(method, f"{PRIMARY_ORIGIN}{path}", headers=headers, params=params, body=body)

    async def primary_async(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Asynchronously send a request to the primary API."""
        headers = {**self._json_headers, "x-api-token": self._api_key}
        return await self._request_async(method, f"{PRIMARY_ORIGIN}{path}", headers=headers, params=params, body=body)

    def serp(self, body: dict[str, Any], *, response_format: Literal["json", "html"]) -> Any:
        """Send a request to the synchronous Google SERP API."""
        headers = {**self._json_headers, "Authorization": f"Bearer {self._api_key}"}
        return self._request(
            "POST",
            f"{SERP_ORIGIN}/api/google/serp/v2/sync",
            headers=headers,
            body=body,
            force_text=response_format == "html",
            force_json=response_format == "json",
        )

    async def serp_async(self, body: dict[str, Any], *, response_format: Literal["json", "html"]) -> Any:
        """Asynchronously send a request to the synchronous Google SERP API."""
        headers = {**self._json_headers, "Authorization": f"Bearer {self._api_key}"}
        return await self._request_async(
            "POST",
            f"{SERP_ORIGIN}/api/google/serp/v2/sync",
            headers=headers,
            body=body,
            force_text=response_format == "html",
            force_json=response_format == "json",
        )

    def rendered(self, *, params: dict[str, Any], body: dict[str, Any]) -> Any:
        """Send a rendered-page request with mandatory query-token authentication."""
        query = {"token": self._api_key, "browserRendering": "true", **params}
        return self._request("POST", f"{RENDERED_ORIGIN}/", headers=self._json_headers, params=query, body=body)

    async def rendered_async(self, *, params: dict[str, Any], body: dict[str, Any]) -> Any:
        """Asynchronously send a rendered-page request with mandatory query-token authentication."""
        query = {"token": self._api_key, "browserRendering": "true", **params}
        return await self._request_async(
            "POST", f"{RENDERED_ORIGIN}/", headers=self._json_headers, params=query, body=body
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        force_text: bool = False,
        force_json: bool = False,
    ) -> Any:
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=False) as client:
                response = client.request(method, url, headers=headers, params=params, json=body)
            response.raise_for_status()
        except httpx.TimeoutException:
            msg = "The MrScraper API request timed out."
            raise MrScraperError(msg) from None
        except httpx.HTTPStatusError as error:
            detail = self._redact(error.response.text)[:_MAX_ERROR_DETAIL]
            msg = f"MrScraper API request failed with status {error.response.status_code}: {detail}"
            raise MrScraperError(msg) from None
        except httpx.HTTPError as error:
            msg = f"MrScraper API request failed: {self._redact(str(error))[:_MAX_ERROR_DETAIL]}"
            raise MrScraperError(msg) from None
        return self._decode(response, force_text=force_text, force_json=force_json)

    async def _request_async(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        force_text: bool = False,
        force_json: bool = False,
    ) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=False) as client:
                response = await client.request(method, url, headers=headers, params=params, json=body)
            response.raise_for_status()
        except httpx.TimeoutException:
            msg = "The MrScraper API request timed out."
            raise MrScraperError(msg) from None
        except httpx.HTTPStatusError as error:
            detail = self._redact(error.response.text)[:_MAX_ERROR_DETAIL]
            msg = f"MrScraper API request failed with status {error.response.status_code}: {detail}"
            raise MrScraperError(msg) from None
        except httpx.HTTPError as error:
            msg = f"MrScraper API request failed: {self._redact(str(error))[:_MAX_ERROR_DETAIL]}"
            raise MrScraperError(msg) from None
        return self._decode(response, force_text=force_text, force_json=force_json)

    @staticmethod
    def encoded_path_segment(value: str) -> str:
        """Encode a user-provided identifier as exactly one URL path segment."""
        return quote(value, safe="")

    def _redact(self, value: str) -> str:
        redacted = value.replace(self._api_key, "[REDACTED]")
        return redacted.replace(quote(self._api_key, safe=""), "[REDACTED]")

    @staticmethod
    def _decode(response: httpx.Response, *, force_text: bool, force_json: bool) -> Any:
        if force_text:
            return response.text
        content_type = response.headers.get("content-type", "").lower()
        if force_json or "application/json" in content_type or "+json" in content_type:
            try:
                return response.json()
            except ValueError:
                msg = "MrScraper returned an invalid JSON response."
                raise MrScraperError(msg) from None
        return response.text
