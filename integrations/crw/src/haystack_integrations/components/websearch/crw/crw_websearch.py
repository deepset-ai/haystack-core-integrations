# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret

from crw import CrwClient

logger = logging.getLogger(__name__)


@component
class CRWWebSearch:
    """
    A component that uses fastCRW to search the web and return results as Haystack Documents.

    This component wraps the fastCRW Search API, enabling web search queries that return
    structured documents with content and links.

    fastCRW is a Firecrawl-compatible web scraper that ships as a single binary. Web search
    is a managed-cloud feature: with no `api_url` the client targets the managed cloud at
    [fastcrw.com](https://fastcrw.com) and reads the key from the `CRW_API_KEY` environment
    variable. Pass `api_url` to point at a self-hosted server.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.crw import CRWWebSearch
    from haystack.utils import Secret

    websearch = CRWWebSearch(
        api_key=Secret.from_env_var("CRW_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("CRW_API_KEY", strict=False),
        api_url: str | None = None,
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the CRWWebSearch component.

        :param api_key:
            API key for fastCRW. Defaults to the `CRW_API_KEY` environment variable.
            Optional for self-hosted servers that run without authentication.
        :param api_url:
            Base URL of a self-hosted fastCRW server, e.g. `http://localhost:3000`.
            Defaults to the managed cloud (`https://fastcrw.com/api`).
        :param top_k:
            Maximum number of results to return.
        :param search_params:
            Additional parameters passed to the fastCRW search API.
            See the [fastCRW API reference](https://fastcrw.com/docs/rest-api)
            for available options. Supported keys include: `sources`.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.top_k = top_k
        self.search_params = search_params
        self._crw_client: CrwClient | None = None

    def warm_up(self) -> None:
        """
        Initialize the fastCRW client.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._crw_client is None:
            self._crw_client = CrwClient(api_url=self.api_url, api_key=self.api_key.resolve_value())

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using fastCRW and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._crw_client is None:
            self.warm_up()
        if self._crw_client is None:
            msg = "CRWWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params = (search_params if search_params is not None else self.search_params or {}).copy()
        if "limit" not in params and self.top_k is not None:
            params["limit"] = self.top_k

        response = self._crw_client.search(query, **params)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert a fastCRW search response to Haystack Documents and links.

        :param response: fastCRW search results (list of result dictionaries).
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in response or []:
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("markdown") or result.get("description", "")

            documents.append(Document(content=content, meta={"title": title, "url": url}))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
