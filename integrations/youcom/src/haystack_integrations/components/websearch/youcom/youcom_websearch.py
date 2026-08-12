# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry

logger = logging.getLogger(__name__)

YOUCOM_KEYED_SEARCH_URL = "https://api.you.com/v1/search"
YOUCOM_KEYLESS_SEARCH_URL = "https://api.you.com/v1/agents/search"

try:
    _VERSION = version("youcom-haystack")
except PackageNotFoundError:  # pragma: no cover
    _VERSION = "0.0.0-dev"

USER_AGENT = f"youcom-haystack/{_VERSION} youdotcom-integration/deepset-ai-haystack-core-integrations"

_RATE_LIMIT_STATUSES = (402, 429)
_UPGRADE_HINT = (
    "You.com keyless free-tier limit reached. Set the YDC_API_KEY environment variable "
    "(or pass api_key) for higher limits — get a free API key at https://you.com/platform."
)


@component
class YouComWebSearch:
    """
    A component that uses the You.com Search API to search the web and return results as Haystack Documents.

    Works with zero configuration: when no API key is available, searches use You.com's keyless
    free tier (rate limited per IP), so getting-started pipelines run without any setup.
    Set the `YDC_API_KEY` environment variable (or pass `api_key`) to use the keyed
    [You.com Search API](https://you.com/docs/api-reference/search/v1-search) with higher limits.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.youcom import YouComWebSearch

    websearch = YouComWebSearch(top_k=5)  # no API key needed to get started
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("YDC_API_KEY", strict=False),
        top_k: int | None = 10,
        freshness: str | None = None,
        country: str | None = None,
        search_lang: str | None = None,
        safesearch: str | None = None,
        extra_params: dict[str, Any] | None = None,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the YouComWebSearch component.

        :param api_key:
            You.com API key. Defaults to the `YDC_API_KEY` environment variable, resolved
            leniently: when the variable is not set, the component uses You.com's keyless
            free tier (rate limited per IP) instead of raising an error.
        :param top_k:
            Maximum number of results to return per section (web, news). Maps to the
            `count` parameter in the You.com API (1-100).
        :param freshness:
            Only return results from within the given window: `"day"`, `"week"`, `"month"`,
            `"year"`, or a date range in the format `"YYYY-MM-DDtoYYYY-MM-DD"`.
        :param country:
            2-letter country code determining the geographical focus of web results (e.g. `"US"`, `"DE"`).
        :param search_lang:
            Language of the returned web results in BCP 47 format (e.g. `"EN"`, `"PT-BR"`).
            Maps to the `language` parameter in the You.com API.
        :param safesearch:
            Content moderation level: `"off"`, `"moderate"`, or `"strict"`.
        :param extra_params:
            Additional query parameters passed directly to the You.com Search API
            (e.g. `{"include_domains": "nytimes.com,bbc.com"}`).
        :param timeout:
            Timeout in seconds for the HTTP request. Defaults to 10.
        :param max_retries:
            Maximum number of retry attempts on transient failures. Defaults to 3.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.freshness = freshness
        self.country = country
        self.search_lang = search_lang
        self.safesearch = safesearch
        self.extra_params = extra_params
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using the You.com Search API and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        url, headers = self._build_request()
        params = self._build_params(query=query, top_k=top_k)

        try:
            response = request_with_retry(
                attempts=self.max_retries,
                method="GET",
                url=url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.HTTPStatusError as error:
            raise self._enrich_error(error, headers) from error
        return self._parse_response(response.json())

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using the You.com Search API and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        url, headers = self._build_request()
        params = self._build_params(query=query, top_k=top_k)

        try:
            response = await async_request_with_retry(
                attempts=self.max_retries,
                method="GET",
                url=url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.HTTPStatusError as error:
            raise self._enrich_error(error, headers) from error
        return self._parse_response(response.json())

    def _build_request(self) -> tuple[str, dict[str, str]]:
        """
        Select the endpoint and headers based on API key availability.

        With a resolvable API key, requests go to the keyed You.com Search API.
        Without one, requests fall back to the keyless free tier.
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        api_key = self.api_key.resolve_value() if self.api_key else None
        if api_key:
            headers["X-API-Key"] = api_key
            return YOUCOM_KEYED_SEARCH_URL, headers
        return YOUCOM_KEYLESS_SEARCH_URL, headers

    def _build_params(self, query: str, top_k: int | None) -> dict[str, Any]:
        effective_top_k = top_k if top_k is not None else self.top_k
        params: dict[str, Any] = {"query": query}
        if effective_top_k is not None:
            params["count"] = effective_top_k
        if self.freshness is not None:
            params["freshness"] = self.freshness
        if self.country is not None:
            params["country"] = self.country
        if self.search_lang is not None:
            params["language"] = self.search_lang
        if self.safesearch is not None:
            params["safesearch"] = self.safesearch
        if self.extra_params:
            params.update(self.extra_params)
        return params

    @staticmethod
    def _enrich_error(error: httpx.HTTPStatusError, headers: dict[str, str]) -> httpx.HTTPStatusError:
        """
        Add upgrade guidance to keyless rate-limit errors; return other errors unchanged.
        """
        if error.response.status_code in _RATE_LIMIT_STATUSES and "X-API-Key" not in headers:
            return httpx.HTTPStatusError(f"{error} — {_UPGRADE_HINT}", request=error.request, response=error.response)
        return error

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a You.com Search API response to Haystack Documents and links.

        Web results come first, then news results. Document content is the joined
        text snippets when available, falling back to the result description.

        :param response: You.com Search API response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        results = response.get("results", {})
        for section in ("web", "news"):
            for result in results.get(section) or []:
                url = result.get("url", "")
                title = result.get("title", "")
                snippets = result.get("snippets") or []
                content = "\n".join(snippets) if snippets else result.get("description", "")

                meta: dict[str, Any] = {"title": title, "url": url, "source": section}
                if result.get("page_age"):
                    meta["page_age"] = result["page_age"]

                documents.append(Document(content=content, meta=meta))
                if url:
                    links.append(url)

        return {"documents": documents, "links": links}
