# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

OUTCRAWL_SEARCH_URL = "https://api.outcrawl.ai/v1/search"


@component
class OutcrawlWebSearch:
    """
    A component that uses Outcrawl to search the web and return results as Haystack Documents.

    This component wraps the Outcrawl Search API, enabling web search queries that return
    structured documents with the result snippet, title and link.

    Outcrawl is web infrastructure for AI agents. You need an Outcrawl API key from
    [outcrawl.ai](https://outcrawl.ai/console); the free plan includes credits every month.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.outcrawl import OutcrawlWebSearch
    from haystack.utils import Secret

    websearch = OutcrawlWebSearch(
        api_key=Secret.from_env_var("OUTCRAWL_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("OUTCRAWL_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the OutcrawlWebSearch component.

        :param api_key:
            API key for Outcrawl. Defaults to the `OUTCRAWL_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. Maps to the `limit` parameter of the Outcrawl Search API.
        :param search_params:
            Additional parameters passed to the Outcrawl Search API request body.
            See the [Outcrawl API reference](https://docs.outcrawl.ai) for the available options.
        :param timeout:
            Request timeout in seconds.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.resolve_value() or ''}",
            "Content-Type": "application/json",
        }

    def _build_body(self, query: str, top_k: int | None, search_params: dict[str, Any] | None) -> dict[str, Any]:
        params = (search_params if search_params is not None else self.search_params or {}).copy()
        effective_top_k = top_k if top_k is not None else self.top_k
        body: dict[str, Any] = {"query": query}
        if effective_top_k is not None:
            body["limit"] = effective_top_k
        body.update({k: v for k, v in params.items() if v is not None})
        return body

    def warm_up(self) -> None:
        """
        Initialize the sync HTTP client.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)

    async def warm_up_async(self) -> None:
        """
        Initialize the async HTTP client on the serving event loop.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)

    def close(self) -> None:
        """
        Release the sync HTTP client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """
        Release the async HTTP client.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Outcrawl and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing the search result snippets.
            - `links`: List of URLs from the search results.
        """
        if self._client is None:
            self.warm_up()

        response = self._client.post(  # type: ignore[union-attr]
            OUTCRAWL_SEARCH_URL,
            headers=self._build_headers(),
            json=self._build_body(query, top_k, search_params),
        )
        response.raise_for_status()
        parsed = self._parse_response(response.json())

        logger.debug(
            "Outcrawl returned {number_documents} documents for the query '{query}'",
            number_documents=len(parsed["documents"]),
            query=query,
        )
        return parsed

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Outcrawl and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing the search result snippets.
            - `links`: List of URLs from the search results.
        """
        if self._async_client is None:
            await self.warm_up_async()

        response = await self._async_client.post(  # type: ignore[union-attr]
            OUTCRAWL_SEARCH_URL,
            headers=self._build_headers(),
            json=self._build_body(query, top_k, search_params),
        )
        response.raise_for_status()
        parsed = self._parse_response(response.json())

        logger.debug(
            "Outcrawl returned {number_documents} documents for the query '{query}'",
            number_documents=len(parsed["documents"]),
            query=query,
        )
        return parsed

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an Outcrawl search response to Haystack Documents and links.

        :param response: Outcrawl search response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for hit in response.get("hits", []):
            url = hit.get("url", "")
            meta: dict[str, Any] = {"title": hit.get("title") or "", "url": url}
            if hit.get("rank") is not None:
                meta["rank"] = hit["rank"]
            documents.append(Document(content=hit.get("snippet") or "", meta=meta))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
