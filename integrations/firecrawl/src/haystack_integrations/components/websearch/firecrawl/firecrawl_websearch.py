# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from firecrawl import AsyncFirecrawl, Firecrawl  # type: ignore[import-untyped]
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)


@component
class FirecrawlWebSearch:
    """
    A component that uses Firecrawl to search the web and return results as Haystack Documents.

    This component wraps the Firecrawl Search API, enabling web search queries that return
    structured documents with content and links. It follows the standard Haystack WebSearch
    component interface.

    Firecrawl is a service that crawls and scrapes websites, returning content in formats suitable
    for LLMs. You need a Firecrawl API key from [firecrawl.dev](https://firecrawl.dev).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.firecrawl import FirecrawlWebSearch
    from haystack.utils import Secret

    websearch = FirecrawlWebSearch(
        api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
        top_k=5,
    )
    websearch.warm_up()

    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("FIRECRAWL_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the FirecrawlWebSearch component.

        :param api_key:
            API key for Firecrawl.
            Defaults to the `FIRECRAWL_API_KEY` environment variable.
        :param top_k:
            Maximum number of documents to return.
            Defaults to 10.
        :param search_params:
            Additional parameters passed to the Firecrawl search API.
            See the [Firecrawl API reference](https://docs.firecrawl.dev/api-reference/endpoint/search)
            for available parameters. Supported keys include: `tbs`, `location`,
            `scrape_options`, `sources`, `categories`, `timeout`.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params
        self._search_params = {} if search_params is None else search_params.copy()
        self._firecrawl_client: Firecrawl | None = None
        self._async_firecrawl_client: AsyncFirecrawl | None = None

    def warm_up(self) -> None:
        """
        Warm up the Firecrawl clients by initializing the sync and async clients.
        This is useful to avoid cold start delays when performing searches.
        """
        if self._firecrawl_client is None:
            self._firecrawl_client = Firecrawl(api_key=self.api_key.resolve_value())
        if self._async_firecrawl_client is None:
            self._async_firecrawl_client = AsyncFirecrawl(api_key=self.api_key.resolve_value())

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            top_k=self.top_k,
            search_params=self.search_params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FirecrawlWebSearch":
        """Deserializes the component from a dictionary."""
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Firecrawl and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional override of search parameters for this run.
            If provided, fully replaces the init-time search_params.
        :returns: A dictionary with the following keys:
            - `documents`: List of documents with search result content.
            - `links`: List of URLs from the search results.
        """
        if self._firecrawl_client is None:
            self.warm_up()

        current_params = search_params if search_params is not None else self._search_params
        params = current_params.copy()
        if "limit" not in params and self.top_k is not None:
            params["limit"] = self.top_k

        try:
            search_response = self._firecrawl_client.search(  # type: ignore[union-attr]
                query=query,
                **params,
            )
        except Exception as error:
            logger.exception(f"Failed to search for query '{query}': {error}")
            return {"documents": [], "links": []}

        documents, links = self._parse_search_response(search_response)

        if self.top_k is not None:
            documents = documents[: self.top_k]
            links = links[: self.top_k]

        return {"documents": documents, "links": links}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Firecrawl and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional override of search parameters for this run.
            If provided, fully replaces the init-time search_params.
        :returns: A dictionary with the following keys:
            - `documents`: List of documents with search result content.
            - `links`: List of URLs from the search results.
        """
        if self._async_firecrawl_client is None:
            self.warm_up()

        current_params = search_params if search_params is not None else self._search_params
        params = current_params.copy()
        if "limit" not in params and self.top_k is not None:
            params["limit"] = self.top_k

        try:
            search_response = await self._async_firecrawl_client.search(  # type: ignore[union-attr]
                query=query,
                **params,
            )
        except Exception as error:
            logger.exception(f"Failed to search for query '{query}': {error}")
            return {"documents": [], "links": []}

        documents, links = self._parse_search_response(search_response)

        if self.top_k is not None:
            documents = documents[: self.top_k]
            links = links[: self.top_k]

        return {"documents": documents, "links": links}

    @staticmethod
    def _parse_search_response(search_response: Any) -> tuple[list[Document], list[str]]:
        """
        Convert a Firecrawl search response to Haystack Documents and links.

        :param search_response: Firecrawl search response object.
        :returns: Tuple of (documents, links).
        """
        documents: list[Document] = []
        links: list[str] = []

        web_results = getattr(search_response, "web", None) or []
        for result in web_results:
            url = ""
            title = ""
            content = ""

            if hasattr(result, "markdown") and result.markdown:
                content = result.markdown
                metadata = result.metadata_dict if hasattr(result, "metadata_dict") else {}
                url = metadata.get("url", getattr(result, "url", ""))
                title = metadata.get("title", "")
            else:
                url = getattr(result, "url", "") or ""
                title = getattr(result, "title", "") or ""
                content = getattr(result, "description", "") or ""

            doc = Document(
                content=content,
                meta={
                    "title": title,
                    "url": url,
                },
            )
            documents.append(doc)
            if url:
                links.append(url)

        return documents, links
