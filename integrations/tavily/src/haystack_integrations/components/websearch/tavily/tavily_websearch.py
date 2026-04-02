# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret

from tavily import AsyncTavilyClient, TavilyClient

logger = logging.getLogger(__name__)


@component
class TavilyWebSearch:
    """
    A component that uses Tavily to search the web and return results as Haystack Documents.

    This component wraps the Tavily Search API, enabling web search queries that return
    structured documents with content and links.

    Tavily is an AI-powered search API optimized for LLM applications. You need a Tavily
    API key from [tavily.com](https://tavily.com).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.tavily import TavilyWebSearch
    from haystack.utils import Secret

    websearch = TavilyWebSearch(
        api_key=Secret.from_env_var("TAVILY_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TAVILY_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the TavilyWebSearch component.

        :param api_key:
            API key for Tavily. Defaults to the `TAVILY_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return.
        :param search_params:
            Additional parameters passed to the Tavily search API.
            See the [Tavily API reference](https://docs.tavily.com/docs/tavily-api/rest_api)
            for available options. Supported keys include: `search_depth`, `include_answer`,
            `include_raw_content`, `include_domains`, `exclude_domains`.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params
        self._tavily_client: TavilyClient | None = None
        self._async_tavily_client: AsyncTavilyClient | None = None

    def warm_up(self) -> None:
        """
        Initialize the Tavily sync and async clients.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._tavily_client is None:
            self._tavily_client = TavilyClient(api_key=self.api_key.resolve_value())
        if self._async_tavily_client is None:
            self._async_tavily_client = AsyncTavilyClient(api_key=self.api_key.resolve_value())

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Tavily and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._tavily_client is None:
            self.warm_up()
        if self._tavily_client is None:
            msg = "TavilyWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params = (search_params if search_params is not None else self.search_params or {}).copy()
        if "max_results" not in params and self.top_k is not None:
            params["max_results"] = self.top_k

        response = self._tavily_client.search(query=query, **params)
        return self._parse_response(response)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Tavily and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._async_tavily_client is None:
            self.warm_up()
        if self._async_tavily_client is None:
            msg = "TavilyWebSearch async client failed to initialize."
            raise RuntimeError(msg)

        params = (search_params if search_params is not None else self.search_params or {}).copy()
        if "max_results" not in params and self.top_k is not None:
            params["max_results"] = self.top_k

        response = await self._async_tavily_client.search(query=query, **params)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a Tavily search response to Haystack Documents and links.

        :param response: Tavily search response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in response.get("results", []):
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", "")
            score = result.get("score")

            documents.append(Document(content=content, meta={"title": title, "url": url}, score=score))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
