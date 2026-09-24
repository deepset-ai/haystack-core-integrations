# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component

from haystack_integrations.components.mrscraper._base import MrScraperComponent
from haystack_integrations.components.mrscraper.extraction import Mode
from haystack_integrations.utils.mrscraper.payloads import (
    build_general_payload,
    build_listing_payload,
    build_map_payload,
)


@component
class MrScraperCreatePromptScraper(MrScraperComponent):
    """Create a reusable General AI scraper from a prompt instead of describing an immediate-only task."""

    @component.output_types(result=Any)
    def run(
        self,
        url: str,
        prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        mode: Mode = "Super",
        proxy_country: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a reusable prompt-based scraper.

        :param url: Nonblank page URL used to create the scraper.
        :param prompt: Optional extraction instructions.
        :param output_schema: Optional JSON shape appended compactly to the prompt.
        :param mode: Scraping mode, `Super` or `Cheap`.
        :param proxy_country: Optional two-letter proxy country code.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = build_general_payload(
            url=url, prompt=prompt, output_schema=output_schema, mode=mode, proxy_country=proxy_country
        )
        return {"result": self._client().primary("POST", "/api/v1/scrapers-ai", body=payload)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        url: str,
        prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        mode: Mode = "Super",
        proxy_country: str | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously create a reusable prompt-based scraper.

        :param url: Nonblank page URL used to create the scraper.
        :param prompt: Optional extraction instructions.
        :param output_schema: Optional JSON shape appended compactly to the prompt.
        :param mode: Scraping mode, `Super` or `Cheap`.
        :param proxy_country: Optional two-letter proxy country code.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = build_general_payload(
            url=url, prompt=prompt, output_schema=output_schema, mode=mode, proxy_country=proxy_country
        )
        result = await self._client().primary_async("POST", "/api/v1/scrapers-ai", body=payload)
        return {"result": result}


@component
class MrScraperCreateListingScraper(MrScraperComponent):
    """Create a reusable Listing AI scraper for repeated or paginated items."""

    @component.output_types(result=Any)
    def run(
        self,
        url: str,
        prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        max_pages: int = 1,
        proxy_country: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a reusable listing scraper.

        :param url: Nonblank starting URL used to create the scraper.
        :param prompt: Optional instructions describing each listing item.
        :param output_schema: Optional JSON item shape appended compactly to the prompt.
        :param max_pages: Maximum pagination pages to scrape. Must be at least 1.
        :param proxy_country: Optional two-letter proxy country code.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = build_listing_payload(
            url=url,
            prompt=prompt,
            output_schema=output_schema,
            max_pages=max_pages,
            proxy_country=proxy_country,
        )
        return {"result": self._client().primary("POST", "/api/v1/scrapers-ai", body=payload)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        url: str,
        prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        max_pages: int = 1,
        proxy_country: str | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously create a reusable listing scraper.

        :param url: Nonblank starting URL used to create the scraper.
        :param prompt: Optional instructions describing each listing item.
        :param output_schema: Optional JSON item shape appended compactly to the prompt.
        :param max_pages: Maximum pagination pages to scrape. Must be at least 1.
        :param proxy_country: Optional two-letter proxy country code.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = build_listing_payload(
            url=url,
            prompt=prompt,
            output_schema=output_schema,
            max_pages=max_pages,
            proxy_country=proxy_country,
        )
        result = await self._client().primary_async("POST", "/api/v1/scrapers-ai", body=payload)
        return {"result": result}


@component
class MrScraperCreateWebsiteCrawlScraper(MrScraperComponent):
    """Create a reusable Map AI scraper for future website URL discovery runs."""

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
        Create a reusable website-crawl scraper.

        :param url: Nonblank starting URL used to create the scraper.
        :param max_depth: Maximum link depth to crawl.
        :param max_pages: Maximum pages to evaluate.
        :param limit: Maximum discovered URLs to return. Must be at least 1.
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
        Asynchronously create a reusable website-crawl scraper.

        :param url: Nonblank starting URL used to create the scraper.
        :param max_depth: Maximum link depth to crawl.
        :param max_pages: Maximum pages to evaluate.
        :param limit: Maximum discovered URLs to return. Must be at least 1.
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
