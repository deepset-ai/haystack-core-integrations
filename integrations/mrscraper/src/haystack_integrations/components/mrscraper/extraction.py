# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component

from haystack_integrations.components.mrscraper._base import MrScraperComponent
from haystack_integrations.utils.mrscraper.payloads import build_general_payload, build_listing_payload
from haystack_integrations.utils.mrscraper.presets import load_structured_data_prompts
from haystack_integrations.utils.mrscraper.validation import (
    optional_nonblank,
    validate_bool,
    validate_choice,
    validate_country_code,
    validate_int,
    validate_nonblank,
)

Mode = Literal["Super", "Cheap"]
StructuredCategory = Literal[
    "article",
    "forumThread",
    "hotel",
    "jobPosting",
    "post",
    "product",
    "property",
    "restaurant",
    "socialMediaProfile",
    "tourAttraction",
]


@component
class MrScraperExtractPageByPrompt(MrScraperComponent):
    """Immediately extract data from one page using a natural-language prompt and optional JSON schema."""

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
        Extract one page with a General AI scraper.

        :param url: Nonblank page URL to extract.
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
        Asynchronously extract one page with a General AI scraper.

        :param url: Nonblank page URL to extract.
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
class MrScraperExtractListings(MrScraperComponent):
    """Immediately extract repeated listings or paginated content with an optional item schema."""

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
        Extract listings or paginated content.

        :param url: Nonblank starting URL to extract.
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
        Asynchronously extract listings or paginated content.

        :param url: Nonblank starting URL to extract.
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
class MrScraperExtractStructuredData(MrScraperComponent):
    """Extract structured data using one of ten exact bundled MrScraper category prompts."""

    @staticmethod
    def _payload(url: str, category: StructuredCategory, mode: Mode, proxy_country: str | None) -> dict[str, Any]:
        category_value = validate_choice(category, "category", tuple(load_structured_data_prompts()))
        payload: dict[str, Any] = {
            "graph": "general",
            "url": validate_nonblank(url, "url"),
            "message": load_structured_data_prompts()[category_value],
            "mode": validate_choice(mode, "mode", ("Super", "Cheap")),
        }
        proxy = validate_country_code(proxy_country, "proxy_country", optional=True)
        if proxy is not None:
            payload["proxyCountry"] = proxy
        return payload

    @component.output_types(result=Any)
    def run(
        self,
        url: str,
        category: StructuredCategory = "article",
        mode: Mode = "Super",
        proxy_country: str | None = None,
    ) -> dict[str, Any]:
        """
        Extract a supported structured-data category.

        :param url: Nonblank page URL to extract.
        :param category: Structured prompt preset category.
        :param mode: Scraping mode, `Super` or `Cheap`.
        :param proxy_country: Optional two-letter proxy country code.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = self._payload(url, category, mode, proxy_country)
        return {"result": self._client().primary("POST", "/api/v1/scrapers-ai", body=payload)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        url: str,
        category: StructuredCategory = "article",
        mode: Mode = "Super",
        proxy_country: str | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously extract a supported structured-data category.

        :param url: Nonblank page URL to extract.
        :param category: Structured prompt preset category.
        :param mode: Scraping mode, `Super` or `Cheap`.
        :param proxy_country: Optional two-letter proxy country code.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        payload = self._payload(url, category, mode, proxy_country)
        result = await self._client().primary_async("POST", "/api/v1/scrapers-ai", body=payload)
        return {"result": result}


@component
class MrScraperFetchRenderedHtml(MrScraperComponent):
    """Fetch a JavaScript-rendered page with browser controls and native JSON or exact text output."""

    @staticmethod
    def _request_data(
        *,
        url: str,
        max_retries: int,
        timeout: int,
        geo_code: str,
        proxy_country: str,
        screenshot: bool | None,
        screenshot_mode: Literal["full", "top"] | None,
        html: bool,
        markdown: bool,
        token_cap: int | None,
        wait_for_selector: str | None,
        wait_until: Literal["domcontentloaded", "load", "networkidle"] | None,
        block_resources: bool | None,
        home_page: bool | None,
        return_cookie: bool | None,
        super_mode: bool | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        screenshot_enabled = screenshot is not None and validate_bool(screenshot, "screenshot")
        query: dict[str, Any] = {
            "timeout": validate_int(timeout, "timeout", minimum=1),
            "geoCode": validate_country_code(geo_code, "geo_code"),
            "html": str(validate_bool(html, "html")).lower(),
            "markdown": str(validate_bool(markdown, "markdown")).lower(),
            "proxyCountry": validate_country_code(proxy_country, "proxy_country"),
        }
        if screenshot_enabled:
            if screenshot_mode is None:
                msg = "'screenshot_mode' must be provided when 'screenshot' is enabled."
                raise ValueError(msg)
            query["screenshot"] = validate_choice(screenshot_mode, "screenshot_mode", ("full", "top"))
        selector = optional_nonblank(wait_for_selector, "wait_for_selector")
        if selector is not None:
            query["waitForSelector"] = selector
        if wait_until is not None:
            query["waitUntil"] = validate_choice(wait_until, "wait_until", ("domcontentloaded", "load", "networkidle"))
        if block_resources is not None and validate_bool(block_resources, "block_resources"):
            query["blockResources"] = "true"
        if return_cookie is not None and validate_bool(return_cookie, "return_cookie"):
            query["returnCookie"] = "true"
        if super_mode is not None and validate_bool(super_mode, "super_mode"):
            query["super"] = "true"

        body: dict[str, Any] = {
            "url": validate_nonblank(url, "url"),
            "maxRetries": validate_int(max_retries, "max_retries", minimum=0),
        }
        if token_cap is not None:
            body["tokenCap"] = validate_int(token_cap, "token_cap", minimum=1)
        if home_page is not None and validate_bool(home_page, "home_page"):
            body["homePage"] = True
        return query, body

    @component.output_types(result=Any)
    def run(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 300,
        geo_code: str = "us",
        proxy_country: str = "us",
        screenshot: bool | None = False,
        screenshot_mode: Literal["full", "top"] | None = None,
        html: bool = True,
        markdown: bool = False,
        token_cap: int | None = None,
        wait_for_selector: str | None = None,
        wait_until: Literal["domcontentloaded", "load", "networkidle"] | None = None,
        block_resources: bool | None = False,
        home_page: bool | None = False,
        return_cookie: bool | None = False,
        super_mode: bool | None = False,
    ) -> dict[str, Any]:
        """
        Fetch a rendered page through the MrScraper stealth browser.

        :param url: Nonblank target URL, sent only as request data.
        :param max_retries: Maximum upstream retry attempts. Must be at least 0.
        :param timeout: Page-load timeout in seconds. Must be at least 1.
        :param geo_code: Two-letter geolocation code.
        :param proxy_country: Two-letter proxy country code.
        :param screenshot: Whether to capture a screenshot.
        :param screenshot_mode: Optional screenshot coverage. Required only when screenshot is enabled.
        :param html: Whether the response should include rendered HTML.
        :param markdown: Whether the response should include converted Markdown.
        :param token_cap: Optional maximum processing token allowance. Must be at least 1 when provided.
        :param wait_for_selector: Optional nonblank CSS selector to await.
        :param wait_until: Optional browser lifecycle event to await.
        :param block_resources: Enable blocking images, fonts, and stylesheets. Omitted when false.
        :param home_page: Enable visiting the site home page first. Omitted when false.
        :param return_cookie: Enable returning browser cookies. Omitted when false.
        :param super_mode: Enable a real device for stronger scraping capabilities. Omitted when false.
        :returns: A dictionary containing native decoded JSON or exact text under `result`.
        """
        query, body = self._request_data(
            url=url,
            max_retries=max_retries,
            timeout=timeout,
            geo_code=geo_code,
            proxy_country=proxy_country,
            screenshot=screenshot,
            screenshot_mode=screenshot_mode,
            html=html,
            markdown=markdown,
            token_cap=token_cap,
            wait_for_selector=wait_for_selector,
            wait_until=wait_until,
            block_resources=block_resources,
            home_page=home_page,
            return_cookie=return_cookie,
            super_mode=super_mode,
        )
        return {"result": self._client().rendered(params=query, body=body)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 300,
        geo_code: str = "us",
        proxy_country: str = "us",
        screenshot: bool | None = False,
        screenshot_mode: Literal["full", "top"] | None = None,
        html: bool = True,
        markdown: bool = False,
        token_cap: int | None = None,
        wait_for_selector: str | None = None,
        wait_until: Literal["domcontentloaded", "load", "networkidle"] | None = None,
        block_resources: bool | None = False,
        home_page: bool | None = False,
        return_cookie: bool | None = False,
        super_mode: bool | None = False,
    ) -> dict[str, Any]:
        """
        Asynchronously fetch a rendered page through the MrScraper stealth browser.

        :param url: Nonblank target URL, sent only as request data.
        :param max_retries: Maximum upstream retry attempts. Must be at least 0.
        :param timeout: Page-load timeout in seconds. Must be at least 1.
        :param geo_code: Two-letter geolocation code.
        :param proxy_country: Two-letter proxy country code.
        :param screenshot: Whether to capture a screenshot.
        :param screenshot_mode: Optional screenshot coverage. Required only when screenshot is enabled.
        :param html: Whether the response should include rendered HTML.
        :param markdown: Whether the response should include converted Markdown.
        :param token_cap: Optional maximum processing token allowance. Must be at least 1 when provided.
        :param wait_for_selector: Optional nonblank CSS selector to await.
        :param wait_until: Optional browser lifecycle event to await.
        :param block_resources: Enable blocking images, fonts, and stylesheets. Omitted when false.
        :param home_page: Enable visiting the site home page first. Omitted when false.
        :param return_cookie: Enable returning browser cookies. Omitted when false.
        :param super_mode: Enable a real device for stronger scraping capabilities. Omitted when false.
        :returns: A dictionary containing native decoded JSON or exact text under `result`.
        """
        query, body = self._request_data(
            url=url,
            max_retries=max_retries,
            timeout=timeout,
            geo_code=geo_code,
            proxy_country=proxy_country,
            screenshot=screenshot,
            screenshot_mode=screenshot_mode,
            html=html,
            markdown=markdown,
            token_cap=token_cap,
            wait_for_selector=wait_for_selector,
            wait_until=wait_until,
            block_resources=block_resources,
            home_page=home_page,
            return_cookie=return_cookie,
            super_mode=super_mode,
        )
        return {"result": await self._client().rendered_async(params=query, body=body)}
