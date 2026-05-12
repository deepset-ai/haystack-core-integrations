# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry

logger = logging.getLogger(__name__)

BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"


@component
class BraveWebSearch:
    """
    A component that uses the Brave Search API to search the web and return results as Haystack Documents.

    You need a Brave Search API key from [brave.com/search/api](https://brave.com/search/api/).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.brave import BraveWebSearch
    from haystack.utils import Secret

    websearch = BraveWebSearch(
        api_key=Secret.from_env_var("BRAVE_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("BRAVE_API_KEY"),
        top_k: int | None = 10,
        country: str | None = None,
        search_lang: str | None = None,
        extra_params: dict[str, Any] | None = None,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the BraveWebSearch component.

        :param api_key:
            Brave Search API key. Defaults to the `BRAVE_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. Maps to the `count` parameter in the Brave API.
        :param country:
            2-letter country code to bias search results (e.g. `"US"`, `"DE"`).
        :param search_lang:
            Language code for search results (e.g. `"en"`, `"de"`).
        :param extra_params:
            Additional query parameters passed directly to the Brave Search API.
        :param timeout:
            Timeout in seconds for the HTTP request. Defaults to 10.
        :param max_retries:
            Maximum number of retry attempts on transient failures. Defaults to 3.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.country = country
        self.search_lang = search_lang
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
        Search the web using Brave Search and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        params = self._build_params(query=query, top_k=top_k)
        headers = self._build_headers()

        response = request_with_retry(
            attempts=self.max_retries,
            method="GET",
            url=BRAVE_SEARCH_API_URL,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Brave Search and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        params = self._build_params(query=query, top_k=top_k)
        headers = self._build_headers()

        response = await async_request_with_retry(
            attempts=self.max_retries,
            method="GET",
            url=BRAVE_SEARCH_API_URL,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _build_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key.resolve_value() or "",
        }

    def _build_params(self, query: str, top_k: int | None) -> dict[str, Any]:
        effective_top_k = top_k if top_k is not None else self.top_k
        params: dict[str, Any] = {"q": query}
        if effective_top_k is not None:
            params["count"] = effective_top_k
        if self.country is not None:
            params["country"] = self.country
        if self.search_lang is not None:
            params["search_lang"] = self.search_lang
        if self.extra_params:
            params.update(self.extra_params)
        return params

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a Brave Search API response to Haystack Documents and links.

        :param response: Brave Search API response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in response.get("web", {}).get("results", []):
            url = result.get("url", "")
            title = result.get("title", "")
            description = result.get("description", "")

            documents.append(Document(content=description, meta={"title": title, "url": url}))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
