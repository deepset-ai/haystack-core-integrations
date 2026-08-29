# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import Document, component
from haystack.utils import Secret

from haystack_integrations.context._client import (
    API_KEY_ENV_VAR,
    DEFAULT_API_URL,
    request_context,
    request_context_async,
)

Freshness = Literal["last_24_hours", "last_week", "last_month", "last_year"]
MAX_RESULTS = 100


@component
class ContextWebSearch:
    """
    Search the live web with Context.dev and return ranked Haystack Documents.

    Context.dev combines relevance-ranked web search with optional Markdown extraction. Create an API key in the
    [Context.dev dashboard](https://www.context.dev/dashboard/api-keys) and set it as `CONTEXT_API_KEY`.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.context import ContextWebSearch

    websearch = ContextWebSearch(top_k=5)
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(API_KEY_ENV_VAR),
        *,
        top_k: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        freshness: Freshness | None = None,
        country: str | None = None,
        include_markdown: bool = False,
        search_params: dict[str, Any] | None = None,
        api_url: str = DEFAULT_API_URL,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Context.dev web search component.

        :param api_key: Context.dev API key. Defaults to the `CONTEXT_API_KEY` environment variable.
        :param top_k: Maximum number of results to return. Must be between 1 and 100.
        :param include_domains: Only return results from these domains.
        :param exclude_domains: Exclude results from these domains.
        :param freshness: Restrict results to a recent time window.
        :param country: Two-letter country code for geographically focused results.
        :param include_markdown: Fetch the full Markdown content for each search result.
        :param search_params: Additional parameters passed to the Context.dev Search API.
        :param api_url: Base URL for the Context.dev API.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum number of retry attempts on transient failures.
        """
        if not 1 <= top_k <= MAX_RESULTS:
            msg = "top_k must be between 1 and 100."
            raise ValueError(msg)
        self.api_key = api_key
        self.top_k = top_k
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.freshness = freshness
        self.country = country
        self.include_markdown = include_markdown
        self.search_params = search_params
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web and return results as Documents.

        :param query: Search query.
        :param top_k: Optional per-run override for the number of results returned.
        :param search_params: Optional per-run replacement for init-time search parameters.
        :returns: A dictionary containing `documents` and `links`.
        """
        effective_top_k = self._effective_top_k(top_k)
        response = request_context(
            api_key=self.api_key,
            api_url=self.api_url,
            method="POST",
            path="/web/search",
            json=self._request_body(query, effective_top_k, search_params),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._parse_response(response, effective_top_k)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web and return results as Documents.

        :param query: Search query.
        :param top_k: Optional per-run override for the number of results returned.
        :param search_params: Optional per-run replacement for init-time search parameters.
        :returns: A dictionary containing `documents` and `links`.
        """
        effective_top_k = self._effective_top_k(top_k)
        response = await request_context_async(
            api_key=self.api_key,
            api_url=self.api_url,
            method="POST",
            path="/web/search",
            json=self._request_body(query, effective_top_k, search_params),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._parse_response(response, effective_top_k)

    def _request_body(
        self,
        query: str,
        top_k: int,
        search_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body = (search_params if search_params is not None else self.search_params or {}).copy()
        optional_params = {
            "includeDomains": self.include_domains,
            "excludeDomains": self.exclude_domains,
            "freshness": self.freshness,
            "country": self.country,
        }
        body.update({name: value for name, value in optional_params.items() if value is not None})
        body.update(
            {
                "query": query,
                "numResults": max(10, top_k),
                "markdownOptions": {
                    "enabled": self.include_markdown,
                    "useMainContentOnly": True,
                    "includeLinks": True,
                    "includeImages": False,
                },
            }
        )
        return body

    def _effective_top_k(self, top_k: int | None) -> int:
        effective_top_k = self.top_k if top_k is None else top_k
        if not 1 <= effective_top_k <= MAX_RESULTS:
            msg = "top_k must be between 1 and 100."
            raise ValueError(msg)
        return effective_top_k

    @staticmethod
    def _parse_response(response: dict[str, Any], top_k: int) -> dict[str, Any]:
        documents: list[Document] = []
        links: list[str] = []
        results = response.get("results", [])
        if not isinstance(results, list):
            return {"documents": documents, "links": links}

        for result in results[:top_k]:
            if not isinstance(result, dict):
                continue
            url = result.get("url", "")
            markdown = result.get("markdown", {})
            markdown_content = markdown.get("markdown") if isinstance(markdown, dict) else None
            content = markdown_content if isinstance(markdown_content, str) else result.get("description", "")
            meta = {
                "title": result.get("title", ""),
                "url": url,
                "relevance": result.get("relevance"),
                "markdown_code": markdown.get("code") if isinstance(markdown, dict) else None,
            }
            documents.append(Document(content=content, meta=meta))
            if isinstance(url, str) and url:
                links.append(url)
        return {"documents": documents, "links": links}
