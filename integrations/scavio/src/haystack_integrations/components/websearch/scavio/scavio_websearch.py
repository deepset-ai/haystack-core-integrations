# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret

from scavio import AsyncScavioClient, ScavioClient

logger = logging.getLogger(__name__)


@component
class ScavioWebSearch:
    """
    A component that uses Scavio to search the web and return results as Haystack Documents.

    This component wraps the Scavio Google web search endpoint, enabling real-time web search
    queries that return structured documents with content and links.

    Scavio is a unified search API for AI agents. You need a Scavio API key from
    [dashboard.scavio.dev](https://dashboard.scavio.dev).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.scavio import ScavioWebSearch
    from haystack.utils import Secret

    websearch = ScavioWebSearch(
        api_key=Secret.from_env_var("SCAVIO_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SCAVIO_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ScavioWebSearch component.

        :param api_key:
            API key for Scavio. Defaults to the `SCAVIO_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return.
        :param search_params:
            Additional parameters passed to the Scavio Google search endpoint. Supported keys
            include: `country_code`, `language`, `page`, `search_type`, `device`, `nfpr`,
            `light_request`.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params
        self._scavio_client: ScavioClient | None = None
        self._async_scavio_client: AsyncScavioClient | None = None

    def warm_up(self) -> None:
        """
        Initialize the Scavio sync and async clients.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._scavio_client is None:
            self._scavio_client = ScavioClient(api_key=self.api_key.resolve_value())
        if self._async_scavio_client is None:
            self._async_scavio_client = AsyncScavioClient(api_key=self.api_key.resolve_value())

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using Scavio and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._scavio_client is None:
            self.warm_up()
        if self._scavio_client is None:
            msg = "ScavioWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params = (search_params if search_params is not None else self.search_params or {}).copy()
        response = self._scavio_client.google.search(query, **params)
        return self._parse_response(response, self.top_k)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using Scavio and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._async_scavio_client is None:
            self.warm_up()
        if self._async_scavio_client is None:
            msg = "ScavioWebSearch async client failed to initialize."
            raise RuntimeError(msg)

        params = (search_params if search_params is not None else self.search_params or {}).copy()
        response = await self._async_scavio_client.google.search(query, **params)
        return self._parse_response(response, self.top_k)

    @staticmethod
    def _parse_response(response: dict[str, Any], top_k: int | None) -> dict[str, Any]:
        """
        Convert a Scavio search response to Haystack Documents and links.

        :param response: Scavio search response dictionary.
        :param top_k: Maximum number of results to keep.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        results = response.get("results", [])
        if top_k is not None:
            results = results[:top_k]

        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", "")

            documents.append(Document(content=content, meta={"title": title, "url": url}))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
