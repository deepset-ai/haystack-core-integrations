# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack_integrations.utils.mrscraper.validation import (
    optional_nonblank,
    validate_bool,
    validate_choice,
    validate_cookie_list,
    validate_country_code,
    validate_dict,
    validate_int,
    validate_integer,
    validate_nonblank,
)

GENERAL_SCHEMA_LABEL = "Return the output as JSON matching this schema:"
LISTING_SCHEMA_LABEL = "Return each item as JSON matching this schema:"


def _with_optional(payload: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        payload[name] = value


def prompt_with_schema(prompt: str | None, output_schema: dict[str, Any] | None, *, label: str) -> str | None:
    """Append a compact JSON schema block to a prompt exactly once."""
    prompt_value = optional_nonblank(prompt, "prompt")
    if output_schema is None:
        return prompt_value
    schema = validate_dict(output_schema, "output_schema")
    schema_block = f"{label}\n{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    if prompt_value is None:
        return schema_block
    if schema_block in prompt_value:
        return prompt_value
    return f"{prompt_value}\n\n{schema_block}"


def build_map_payload(
    *,
    url: str,
    max_depth: int,
    max_pages: int,
    limit: int,
    include_patterns: str | None,
    exclude_patterns: str | None,
) -> dict[str, Any]:
    """Build a validated Map scraper payload."""
    payload: dict[str, Any] = {
        "graph": "map",
        "url": validate_nonblank(url, "url"),
        "maxDepth": validate_integer(max_depth, "max_depth"),
        "maxPages": validate_integer(max_pages, "max_pages"),
        "limit": validate_int(limit, "limit", minimum=1),
    }
    _with_optional(payload, "includePatterns", optional_nonblank(include_patterns, "include_patterns"))
    _with_optional(payload, "excludePatterns", optional_nonblank(exclude_patterns, "exclude_patterns"))
    return payload


def build_general_payload(
    *,
    url: str,
    prompt: str | None,
    output_schema: dict[str, Any] | None,
    mode: str,
    proxy_country: str | None,
) -> dict[str, Any]:
    """Build a validated General scraper payload."""
    payload: dict[str, Any] = {
        "graph": "general",
        "url": validate_nonblank(url, "url"),
        "mode": validate_choice(mode, "mode", ("Super", "Cheap")),
    }
    _with_optional(
        payload,
        "message",
        prompt_with_schema(prompt, output_schema, label=GENERAL_SCHEMA_LABEL),
    )
    _with_optional(payload, "proxyCountry", validate_country_code(proxy_country, "proxy_country", optional=True))
    return payload


def build_listing_payload(
    *,
    url: str,
    prompt: str | None,
    output_schema: dict[str, Any] | None,
    max_pages: int,
    proxy_country: str | None,
) -> dict[str, Any]:
    """Build a validated Listing scraper payload."""
    payload: dict[str, Any] = {
        "graph": "listing",
        "url": validate_nonblank(url, "url"),
        "maxPages": validate_int(max_pages, "max_pages", minimum=1),
    }
    _with_optional(
        payload,
        "message",
        prompt_with_schema(prompt, output_schema, label=LISTING_SCHEMA_LABEL),
    )
    _with_optional(payload, "proxyCountry", validate_country_code(proxy_country, "proxy_country", optional=True))
    return payload


def build_ai_run_payload(
    *,
    scraper_id: str,
    url: str,
    max_retry: int,
    proxy_country: str | None,
    agent_type: str,
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
) -> dict[str, Any]:
    """Build a validated conditional AI rerun payload."""
    agent = validate_choice(agent_type, "agent_type", ("general", "listing", "map"))
    payload: dict[str, Any] = {
        "scraperId": validate_nonblank(scraper_id, "scraper_id"),
        "url": validate_nonblank(url, "url"),
        "maxRetry": validate_int(max_retry, "max_retry", minimum=0),
    }
    _with_optional(payload, "proxyCountry", validate_country_code(proxy_country, "proxy_country", optional=True))

    browser_values = {
        "bypassProxy": bypass_proxy,
        "html": html,
        "markdown": markdown,
        "renderJavascript": render_javascript,
        "returnCookies": return_cookies,
        "screenshot": screenshot,
        "useHomePage": use_home_page,
    }
    map_values = {
        "max_depth": max_depth,
        "limit": limit,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
    }

    if agent == "map":
        incompatible = [name for name, value in browser_values.items() if value is not None]
        incompatible.extend(name for name, value in {"timeout": timeout, "stream": stream}.items() if value is not None)
        if wait_for_selector is not None:
            incompatible.append("wait_for_selector")
        if incompatible:
            msg = f"Map AI runs do not accept: {', '.join(sorted(incompatible))}."
            raise ValueError(msg)
        if max_depth is not None:
            payload["maxDepth"] = validate_int(max_depth, "max_depth", minimum=0)
        if max_pages is not None:
            payload["maxPages"] = validate_int(max_pages, "max_pages", minimum=1)
        if limit is not None:
            payload["limit"] = validate_int(limit, "limit", minimum=1)
        _with_optional(payload, "includePatterns", optional_nonblank(include_patterns, "include_patterns"))
        _with_optional(payload, "excludePatterns", optional_nonblank(exclude_patterns, "exclude_patterns"))
        return payload

    incompatible = [name for name, value in map_values.items() if value is not None]
    if incompatible:
        msg = f"{agent.title()} AI runs do not accept: {', '.join(sorted(incompatible))}."
        raise ValueError(msg)

    if agent == "general" and any(value is not None for value in (max_pages, timeout, stream)):
        msg = "General AI runs do not accept 'max_pages', 'timeout', or 'stream'."
        raise ValueError(msg)

    if agent == "listing":
        if max_pages is not None:
            payload["maxPages"] = validate_int(max_pages, "max_pages", minimum=1)
        if timeout is not None:
            payload["timeout"] = validate_int(timeout, "timeout", minimum=1)
        if stream is not None:
            payload["stream"] = validate_bool(stream, "stream")

    for api_name, value in browser_values.items():
        if value is not None:
            payload[api_name] = validate_bool(value, api_name)
    _with_optional(payload, "waitForSelector", optional_nonblank(wait_for_selector, "wait_for_selector"))
    return payload


def build_manual_run_payload(
    *,
    scraper_id: str,
    url: str,
    max_retry: int,
    proxy_country: str | None,
    bypass_proxy: bool | None,
    cookie_jar: str | None,
    cookies: list[dict[str, Any]] | None,
    home_page: bool | None,
    home_page_timeout: int | None,
    html: bool | None,
    markdown: bool | None,
    paginator: dict[str, Any] | None,
    proxy: str | None,
    record: bool | None,
    return_cookie: bool | None,
    screenshot: bool | None,
    stream: bool | None,
    timeout: int | None,
    token_cap: int | None,
) -> dict[str, Any]:
    """Build a validated Manual scraper rerun payload."""
    payload: dict[str, Any] = {
        "scraperId": validate_nonblank(scraper_id, "scraper_id"),
        "url": validate_nonblank(url, "url"),
        "maxRetry": validate_int(max_retry, "max_retry", minimum=0),
    }
    bool_values = {
        "bypassProxy": (bypass_proxy, "bypass_proxy"),
        "homePage": (home_page, "home_page"),
        "html": (html, "html"),
        "markdown": (markdown, "markdown"),
        "record": (record, "record"),
        "returnCookie": (return_cookie, "return_cookie"),
        "stream": (stream, "stream"),
    }
    for api_name, (value, parameter_name) in bool_values.items():
        if value is not None:
            payload[api_name] = validate_bool(value, parameter_name)
    if cookies is not None:
        payload["cookies"] = validate_cookie_list(cookies)
    if home_page_timeout is not None:
        payload["homePageTimeout"] = validate_int(home_page_timeout, "home_page_timeout", minimum=1)
    if paginator is not None:
        payload["paginator"] = validate_dict(paginator, "paginator")
    if screenshot is not None:
        payload["screenshot"] = "true" if validate_bool(screenshot, "screenshot") else "false"
    if timeout is not None:
        payload["timeout"] = validate_int(timeout, "timeout", minimum=1)
    if token_cap is not None:
        payload["tokenCap"] = validate_int(token_cap, "token_cap", minimum=0)
    _with_optional(payload, "proxyCountry", validate_country_code(proxy_country, "proxy_country", optional=True))
    _with_optional(payload, "cookieJar", optional_nonblank(cookie_jar, "cookie_jar"))
    _with_optional(payload, "proxy", optional_nonblank(proxy, "proxy"))
    return payload
