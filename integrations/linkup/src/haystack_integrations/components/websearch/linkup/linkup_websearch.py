# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import Document, component, logging
from haystack.utils import Secret

from linkup import LinkupClient

logger = logging.getLogger(__name__)


@component
class LinkupWebSearch:
    """
    A component that uses Linkup to search the web and return results as Haystack Documents.

    This component wraps the Linkup Search API, enabling web search queries that return
    structured documents with content and links.

    Linkup is a web search API optimized for LLM applications. You need a Linkup API key
    from [linkup.so](https://www.linkup.so).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.linkup import LinkupWebSearch
    from haystack.utils import Secret

    websearch = LinkupWebSearch(
        api_key=Secret.from_env_var("LINKUP_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("LINKUP_API_KEY"),
        top_k: int | None = 10,
        depth: Literal["fast", "standard", "deep"] = "standard",
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the LinkupWebSearch component.

        :param api_key:
            API key for Linkup. Defaults to the `LINKUP_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. Maps to the `max_results` parameter of the Linkup API.
        :param depth:
            The depth of the search. Can be `"fast"` (beta, sub-second, keyword-based queries only),
            `"standard"` for a simple search, or `"deep"` for a more powerful agentic workflow.
        :param search_params:
            Additional parameters passed to the Linkup search API.
            See the [Linkup API reference](https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search)
            for available options. Supported keys include: `include_images`, `from_date`, `to_date`,
            `include_domains`, `exclude_domains`.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.depth = depth
        self.search_params = search_params
        self._client: LinkupClient | None = None

    def warm_up(self) -> None:
        """
        Initialize the Linkup client.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._client is None:
            self._client = LinkupClient(api_key=self.api_key.resolve_value())

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Linkup and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._client is None:
            self.warm_up()
        if self._client is None:
            msg = "LinkupWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params = self._build_params(search_params)
        response = self._client.search(query=query, depth=self.depth, output_type="searchResults", **params)
        return self._parse_response(response)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Linkup and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._client is None:
            self.warm_up()
        if self._client is None:
            msg = "LinkupWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params = self._build_params(search_params)
        response = await self._client.async_search(query=query, depth=self.depth, output_type="searchResults", **params)
        return self._parse_response(response)

    def _build_params(self, search_params: dict[str, Any] | None) -> dict[str, Any]:
        params = (search_params if search_params is not None else self.search_params or {}).copy()
        if "max_results" not in params and self.top_k is not None:
            params["max_results"] = self.top_k
        return params

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        """
        Convert a Linkup `searchResults` response to Haystack Documents and links.

        :param response: Linkup `LinkupSearchResults` response object.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in getattr(response, "results", []):
            url = getattr(result, "url", "")
            title = getattr(result, "name", "")
            # Text results carry `content`; image results do not.
            content = getattr(result, "content", "")

            documents.append(Document(content=content, meta={"title": title, "url": url}))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
