# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component

from haystack_integrations.components.mrscraper._base import MrScraperComponent
from haystack_integrations.utils.mrscraper.payloads import build_ai_run_payload, build_manual_run_payload
from haystack_integrations.utils.mrscraper.validation import validate_choice, validate_nonblank, validate_urls

ScraperType = Literal["ai", "manual"]
AgentType = Literal["general", "listing", "map"]

_RUN_COMMON_FIELDS = ("scraper_type", "scraper_id", "url", "max_retry", "proxy_country")
_AI_BROWSER_FIELDS = (
    "bypass_proxy",
    "html",
    "markdown",
    "render_javascript",
    "return_cookies",
    "screenshot",
    "use_home_page",
    "wait_for_selector",
)
_MANUAL_FIELDS = (
    "timeout",
    "bypass_proxy",
    "html",
    "markdown",
    "screenshot",
    "stream",
    "cookie_jar",
    "cookies",
    "home_page",
    "home_page_timeout",
    "paginator",
    "proxy",
    "record",
    "return_cookie",
    "token_cap",
)


def conditional_run_tool_schema(generated_schema: dict[str, Any]) -> dict[str, Any]:
    """Turn the flat component schema into Manual and agent-specific AI branches for Agent tools."""
    generated_properties = generated_schema["properties"]

    def discriminator(name: str, value: str) -> dict[str, Any]:
        return {
            "const": value,
            "description": generated_properties[name]["description"],
            "type": "string",
        }

    def branch(
        title: str,
        *,
        scraper_type: ScraperType,
        fields: tuple[str, ...],
        agent_type: AgentType | None = None,
    ) -> dict[str, Any]:
        names = (*_RUN_COMMON_FIELDS, *fields)
        properties = {name: generated_properties[name] for name in names}
        properties["scraper_type"] = discriminator("scraper_type", scraper_type)
        required = ["scraper_type", "scraper_id", "url"]
        if agent_type is not None:
            properties["agent_type"] = discriminator("agent_type", agent_type)
            required.append("agent_type")
        return {
            "title": title,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    branches = [
        branch("Manual scraper", scraper_type="manual", fields=_MANUAL_FIELDS),
        branch("General AI scraper", scraper_type="ai", agent_type="general", fields=_AI_BROWSER_FIELDS),
        branch(
            "Listing AI scraper",
            scraper_type="ai",
            agent_type="listing",
            fields=(*_AI_BROWSER_FIELDS, "max_pages", "timeout", "stream"),
        ),
        branch(
            "Map AI scraper",
            scraper_type="ai",
            agent_type="map",
            fields=("max_pages", "max_depth", "limit", "include_patterns", "exclude_patterns"),
        ),
    ]
    return {
        "type": "object",
        "properties": {name: generated_properties[name] for name in _RUN_COMMON_FIELDS},
        "required": ["scraper_type", "scraper_id", "url"],
        "oneOf": branches,
    }


def _provided(values: dict[str, Any]) -> list[str]:
    return [name for name, value in values.items() if value is not None]


@component
class MrScraperRunExistingScraper(MrScraperComponent):
    """Run one URL with an existing AI or manual scraper using strict type-specific options."""

    @staticmethod
    def _payload(
        *,
        scraper_type: ScraperType,
        scraper_id: str,
        url: str,
        max_retry: int,
        proxy_country: str | None,
        agent_type: AgentType | None,
        max_pages: int | None,
        timeout: int | None,
        bypass_proxy: bool | None,
        html: bool | None,
        markdown: bool | None,
        render_javascript: bool | None,
        return_cookies: bool | None,
        screenshot: bool | None,
        stream: bool | None,
        use_home_page: bool | None,
        wait_for_selector: str | None,
        max_depth: int | None,
        limit: int | None,
        include_patterns: str | None,
        exclude_patterns: str | None,
        cookie_jar: str | None,
        cookies: list[dict[str, Any]] | None,
        home_page: bool | None,
        home_page_timeout: int | None,
        paginator: dict[str, Any] | None,
        proxy: str | None,
        record: bool | None,
        return_cookie: bool | None,
        token_cap: int | None,
    ) -> tuple[str, dict[str, Any]]:
        selected_type = validate_choice(scraper_type, "scraper_type", ("ai", "manual"))
        if selected_type == "manual":
            incompatible = _provided(
                {
                    "agent_type": agent_type,
                    "max_pages": max_pages,
                    "render_javascript": render_javascript,
                    "return_cookies": return_cookies,
                    "use_home_page": use_home_page,
                    "wait_for_selector": wait_for_selector,
                    "max_depth": max_depth,
                    "limit": limit,
                    "include_patterns": include_patterns,
                    "exclude_patterns": exclude_patterns,
                }
            )
            if incompatible:
                msg = f"Manual scraper runs do not accept AI-only options: {', '.join(sorted(incompatible))}."
                raise ValueError(msg)
            payload = build_manual_run_payload(
                scraper_id=scraper_id,
                url=url,
                max_retry=max_retry,
                proxy_country=proxy_country,
                bypass_proxy=bypass_proxy,
                cookie_jar=cookie_jar,
                cookies=cookies,
                home_page=home_page,
                home_page_timeout=home_page_timeout,
                html=html,
                markdown=markdown,
                paginator=paginator,
                proxy=proxy,
                record=record,
                return_cookie=return_cookie,
                screenshot=screenshot,
                stream=stream,
                timeout=timeout,
                token_cap=token_cap,
            )
            return "/api/v1/scrapers-manual-rerun", payload

        manual_options = _provided(
            {
                "cookie_jar": cookie_jar,
                "cookies": cookies,
                "home_page": home_page,
                "home_page_timeout": home_page_timeout,
                "paginator": paginator,
                "proxy": proxy,
                "record": record,
                "return_cookie": return_cookie,
                "token_cap": token_cap,
            }
        )
        if manual_options:
            msg = f"AI scraper runs do not accept manual-only options: {', '.join(sorted(manual_options))}."
            raise ValueError(msg)
        payload = build_ai_run_payload(
            scraper_id=scraper_id,
            url=url,
            max_retry=max_retry,
            proxy_country=proxy_country,
            agent_type="general" if agent_type is None else agent_type,
            max_pages=max_pages,
            timeout=timeout,
            bypass_proxy=bypass_proxy,
            html=html,
            markdown=markdown,
            render_javascript=render_javascript,
            return_cookies=return_cookies,
            screenshot=screenshot,
            stream=stream,
            use_home_page=use_home_page,
            wait_for_selector=wait_for_selector,
            max_depth=max_depth,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        return "/api/v1/scrapers-ai-rerun", payload

    @component.output_types(result=Any)
    def run(
        self,
        scraper_type: ScraperType,
        scraper_id: str,
        url: str,
        max_retry: int = 3,
        proxy_country: str | None = None,
        agent_type: AgentType | None = None,
        max_pages: int | None = None,
        timeout: int | None = None,
        bypass_proxy: bool | None = None,
        html: bool | None = None,
        markdown: bool | None = None,
        render_javascript: bool | None = None,
        return_cookies: bool | None = None,
        screenshot: bool | None = None,
        stream: bool | None = None,
        use_home_page: bool | None = None,
        wait_for_selector: str | None = None,
        max_depth: int | None = None,
        limit: int | None = None,
        include_patterns: str | None = None,
        exclude_patterns: str | None = None,
        cookie_jar: str | None = None,
        cookies: list[dict[str, Any]] | None = None,
        home_page: bool | None = None,
        home_page_timeout: int | None = None,
        paginator: dict[str, Any] | None = None,
        proxy: str | None = None,
        record: bool | None = None,
        return_cookie: bool | None = None,
        token_cap: int | None = None,
    ) -> dict[str, Any]:
        """
        Run one URL with an existing scraper.

        `None` on an advanced setting means it is omitted so the API can apply its own default.

        :param scraper_type: Existing scraper type, `ai` or `manual`.
        :param scraper_id: Nonblank ID of the existing scraper.
        :param url: Nonblank URL to process.
        :param max_retry: Maximum retry attempts. Must be at least 0.
        :param proxy_country: Optional two-letter proxy country code.
        :param agent_type: AI agent type; defaults to `general` for direct component use and must be absent for Manual.
        :param max_pages: Optional Listing or Map page limit; valid only for those AI agent types.
        :param timeout: Optional Listing or Manual timeout; invalid for General and Map AI.
        :param bypass_proxy: Optional General/Listing AI or Manual proxy-bypass setting.
        :param html: Whether to include HTML; valid for General/Listing AI and Manual.
        :param markdown: Whether to include Markdown; valid for General/Listing AI and Manual.
        :param render_javascript: Whether General/Listing AI should render JavaScript.
        :param return_cookies: Whether General/Listing AI should return cookies.
        :param screenshot: Whether to capture a screenshot; Manual sends lowercase string booleans.
        :param stream: Whether to stream results; valid for Listing AI and Manual.
        :param use_home_page: Whether General/Listing AI should first visit the home page.
        :param wait_for_selector: Optional CSS selector for General/Listing AI.
        :param max_depth: Optional Map AI crawl depth with minimum 0.
        :param limit: Optional Map AI result limit with minimum 1.
        :param include_patterns: Optional Map AI pipe-separated include patterns.
        :param exclude_patterns: Optional Map AI pipe-separated exclude patterns.
        :param cookie_jar: Optional Manual cookie-jar identifier or serialized value.
        :param cookies: Optional Manual browser cookies as a list of dictionaries.
        :param home_page: Whether Manual should first visit the home page.
        :param home_page_timeout: Optional Manual home-page timeout with minimum 1.
        :param paginator: Optional Manual pagination configuration dictionary.
        :param proxy: Optional Manual proxy URL.
        :param record: Whether to record the Manual browser session.
        :param return_cookie: Whether Manual should return browser cookies.
        :param token_cap: Optional Manual result token cap with minimum 0.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        path, payload = self._payload(
            scraper_type=scraper_type,
            scraper_id=scraper_id,
            url=url,
            max_retry=max_retry,
            proxy_country=proxy_country,
            agent_type=agent_type,
            max_pages=max_pages,
            timeout=timeout,
            bypass_proxy=bypass_proxy,
            html=html,
            markdown=markdown,
            render_javascript=render_javascript,
            return_cookies=return_cookies,
            screenshot=screenshot,
            stream=stream,
            use_home_page=use_home_page,
            wait_for_selector=wait_for_selector,
            max_depth=max_depth,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            cookie_jar=cookie_jar,
            cookies=cookies,
            home_page=home_page,
            home_page_timeout=home_page_timeout,
            paginator=paginator,
            proxy=proxy,
            record=record,
            return_cookie=return_cookie,
            token_cap=token_cap,
        )
        return {"result": self._client().primary("POST", path, body=payload)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        scraper_type: ScraperType,
        scraper_id: str,
        url: str,
        max_retry: int = 3,
        proxy_country: str | None = None,
        agent_type: AgentType | None = None,
        max_pages: int | None = None,
        timeout: int | None = None,
        bypass_proxy: bool | None = None,
        html: bool | None = None,
        markdown: bool | None = None,
        render_javascript: bool | None = None,
        return_cookies: bool | None = None,
        screenshot: bool | None = None,
        stream: bool | None = None,
        use_home_page: bool | None = None,
        wait_for_selector: str | None = None,
        max_depth: int | None = None,
        limit: int | None = None,
        include_patterns: str | None = None,
        exclude_patterns: str | None = None,
        cookie_jar: str | None = None,
        cookies: list[dict[str, Any]] | None = None,
        home_page: bool | None = None,
        home_page_timeout: int | None = None,
        paginator: dict[str, Any] | None = None,
        proxy: str | None = None,
        record: bool | None = None,
        return_cookie: bool | None = None,
        token_cap: int | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously run one URL with an existing scraper.

        `None` on an advanced setting means it is omitted so the API can apply its own default.

        :param scraper_type: Existing scraper type, `ai` or `manual`.
        :param scraper_id: Nonblank ID of the existing scraper.
        :param url: Nonblank URL to process.
        :param max_retry: Maximum retry attempts. Must be at least 0.
        :param proxy_country: Optional two-letter proxy country code.
        :param agent_type: AI agent type; defaults to `general` for direct component use and must be absent for Manual.
        :param max_pages: Optional Listing or Map page limit; valid only for those AI agent types.
        :param timeout: Optional Listing or Manual timeout; invalid for General and Map AI.
        :param bypass_proxy: Optional General/Listing AI or Manual proxy-bypass setting.
        :param html: Whether to include HTML; valid for General/Listing AI and Manual.
        :param markdown: Whether to include Markdown; valid for General/Listing AI and Manual.
        :param render_javascript: Whether General/Listing AI should render JavaScript.
        :param return_cookies: Whether General/Listing AI should return cookies.
        :param screenshot: Whether to capture a screenshot; Manual sends lowercase string booleans.
        :param stream: Whether to stream results; valid for Listing AI and Manual.
        :param use_home_page: Whether General/Listing AI should first visit the home page.
        :param wait_for_selector: Optional CSS selector for General/Listing AI.
        :param max_depth: Optional Map AI crawl depth with minimum 0.
        :param limit: Optional Map AI result limit with minimum 1.
        :param include_patterns: Optional Map AI pipe-separated include patterns.
        :param exclude_patterns: Optional Map AI pipe-separated exclude patterns.
        :param cookie_jar: Optional Manual cookie-jar identifier or serialized value.
        :param cookies: Optional Manual browser cookies as a list of dictionaries.
        :param home_page: Whether Manual should first visit the home page.
        :param home_page_timeout: Optional Manual home-page timeout with minimum 1.
        :param paginator: Optional Manual pagination configuration dictionary.
        :param proxy: Optional Manual proxy URL.
        :param record: Whether to record the Manual browser session.
        :param return_cookie: Whether Manual should return browser cookies.
        :param token_cap: Optional Manual result token cap with minimum 0.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        path, payload = self._payload(
            scraper_type=scraper_type,
            scraper_id=scraper_id,
            url=url,
            max_retry=max_retry,
            proxy_country=proxy_country,
            agent_type=agent_type,
            max_pages=max_pages,
            timeout=timeout,
            bypass_proxy=bypass_proxy,
            html=html,
            markdown=markdown,
            render_javascript=render_javascript,
            return_cookies=return_cookies,
            screenshot=screenshot,
            stream=stream,
            use_home_page=use_home_page,
            wait_for_selector=wait_for_selector,
            max_depth=max_depth,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            cookie_jar=cookie_jar,
            cookies=cookies,
            home_page=home_page,
            home_page_timeout=home_page_timeout,
            paginator=paginator,
            proxy=proxy,
            record=record,
            return_cookie=return_cookie,
            token_cap=token_cap,
        )
        return {"result": await self._client().primary_async("POST", path, body=payload)}


@component
class MrScraperRunExistingScraperBatch(MrScraperComponent):
    """Run multiple URLs in one batch with an existing AI or manual scraper."""

    @staticmethod
    def _request_data(scraper_type: ScraperType, scraper_id: str, urls: list[str]) -> tuple[str, dict[str, Any]]:
        selected_type = validate_choice(scraper_type, "scraper_type", ("ai", "manual"))
        base = "scrapers-ai-rerun" if selected_type == "ai" else "scrapers-manual-rerun"
        path = f"/api/v1/{base}/bulk"
        return path, {"scraperId": validate_nonblank(scraper_id, "scraper_id"), "urls": validate_urls(urls)}

    @component.output_types(result=Any)
    def run(self, scraper_type: ScraperType, scraper_id: str, urls: list[str]) -> dict[str, Any]:
        """
        Run an existing scraper for a batch of URLs.

        :param scraper_type: Existing scraper type, `ai` or `manual`.
        :param scraper_id: Nonblank ID of the existing scraper.
        :param urls: Nonempty list of nonblank URL strings.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        path, payload = self._request_data(scraper_type, scraper_id, urls)
        return {"result": self._client().primary("POST", path, body=payload)}

    @component.output_types(result=Any)
    async def run_async(self, scraper_type: ScraperType, scraper_id: str, urls: list[str]) -> dict[str, Any]:
        """
        Asynchronously run an existing scraper for a batch of URLs.

        :param scraper_type: Existing scraper type, `ai` or `manual`.
        :param scraper_id: Nonblank ID of the existing scraper.
        :param urls: Nonempty list of nonblank URL strings.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        path, payload = self._request_data(scraper_type, scraper_id, urls)
        return {"result": await self._client().primary_async("POST", path, body=payload)}
