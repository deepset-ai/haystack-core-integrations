# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component

from haystack_integrations.components.mrscraper._base import MrScraperComponent
from haystack_integrations.utils.mrscraper.payloads import build_map_payload
from haystack_integrations.utils.mrscraper.validation import (
    validate_bool,
    validate_choice,
    validate_country_code,
    validate_int,
    validate_nonblank,
)


@component
class MrScraperCrawlWebsiteUrls(MrScraperComponent):
    """Discover URLs by crawling links from one starting website immediately."""

    @component.output_types(result=Any)
    def run(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 50,
        include_patterns: str | None = None,
        exclude_patterns: str | None = None,
    ) -> dict[str, Any]:
        """
        Crawl a website for URLs.

        :param url: Nonblank starting URL to crawl.
        :param max_depth: Maximum link depth to crawl.
        :param max_pages: Maximum number of pages to evaluate.
        :param limit: Maximum number of discovered URLs to return. Must be at least 1.
        :param include_patterns: Optional pipe-separated regular expressions for URLs to include.
        :param exclude_patterns: Optional pipe-separated regular expressions for URLs to exclude.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = build_map_payload(
            url=url,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        return {"result": self._client().primary("POST", "/api/v1/scrapers-ai", body=payload)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 50,
        include_patterns: str | None = None,
        exclude_patterns: str | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously crawl a website for URLs.

        :param url: Nonblank starting URL to crawl.
        :param max_depth: Maximum link depth to crawl.
        :param max_pages: Maximum number of pages to evaluate.
        :param limit: Maximum number of discovered URLs to return. Must be at least 1.
        :param include_patterns: Optional pipe-separated regular expressions for URLs to include.
        :param exclude_patterns: Optional pipe-separated regular expressions for URLs to exclude.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = build_map_payload(
            url=url,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        result = await self._client().primary_async("POST", "/api/v1/scrapers-ai", body=payload)
        return {"result": result}


@component
class MrScraperSearchGoogleSerp(MrScraperComponent):
    """Search Google synchronously and return native JSON results or exact HTML text."""

    @staticmethod
    def _payload(
        query: str,
        region: str,
        language: str,
        page: int,
        format: Literal["json", "html"],  # noqa: A002
        render_js: bool,
    ) -> dict[str, Any]:
        return {
            "query": validate_nonblank(query, "query"),
            "region": validate_country_code(region, "region"),
            "language": validate_country_code(language, "language"),
            "page": validate_int(page, "page", minimum=1),
            "format": validate_choice(format, "format", ("json", "html")),
            "renderJs": validate_bool(render_js, "render_js"),
        }

    @component.output_types(result=Any)
    def run(
        self,
        query: str,
        region: str = "us",
        language: str = "en",
        page: int = 1,
        format: Literal["json", "html"] = "json",  # noqa: A002
        render_js: bool = False,
    ) -> dict[str, Any]:
        """
        Search Google through the MrScraper SERP v2 API.

        :param query: Nonblank Google search query.
        :param region: Two-letter country or region code.
        :param language: Two-letter result language code.
        :param page: Results page number. Must be at least 1.
        :param format: Response format: `json` or `html`.
        :param render_js: Whether to render JavaScript before collecting results.
        :returns: Native decoded JSON for JSON format, or exact response text for HTML format, under `result`.
        """
        payload = self._payload(query, region, language, page, format, render_js)
        return {"result": self._client().serp(payload, response_format=format)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        query: str,
        region: str = "us",
        language: str = "en",
        page: int = 1,
        format: Literal["json", "html"] = "json",  # noqa: A002
        render_js: bool = False,
    ) -> dict[str, Any]:
        """
        Asynchronously search Google through the MrScraper SERP v2 API.

        :param query: Nonblank Google search query.
        :param region: Two-letter country or region code.
        :param language: Two-letter result language code.
        :param page: Results page number. Must be at least 1.
        :param format: Response format: `json` or `html`.
        :param render_js: Whether to render JavaScript before collecting results.
        :returns: Native decoded JSON for JSON format, or exact response text for HTML format, under `result`.
        """
        payload = self._payload(query, region, language, page, format, render_js)
        return {"result": await self._client().serp_async(payload, response_format=format)}
