# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from haystack import Document, component, logging

from ddgs import DDGS

logger = logging.getLogger(__name__)


@component
class DDGSWebSearch:
    """
    Searches the web with ddgs (Dux Distributed Global Search) and returns results as Haystack Documents.

    [ddgs](https://github.com/deedy5/ddgs) is a free, **keyless** metasearch library that aggregates
    results from multiple backends (DuckDuckGo, Google, Bing, Brave, Yahoo, Yandex, Mullvad, and more),
    so no API key is required.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.ddgs import DDGSWebSearch

    websearch = DDGSWebSearch(top_k=5)
    result = websearch.run(query="What is Haystack by deepset?")

    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        top_k: int = 10,
        backend: str = "auto",
        region: str = "us-en",
        safesearch: str = "moderate",
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the DDGSWebSearch component.

        :param top_k:
            Maximum number of results to return.
        :param backend:
            Comma-separated ddgs backends to query, or ``"auto"`` to let ddgs choose
            (for example ``"duckduckgo, google, brave"``). See the ddgs docs for the full list.
        :param region:
            Region/locale for the search, for example ``"us-en"``, ``"de-de"``, or ``"wt-wt"`` (no region).
        :param safesearch:
            Safe-search level: ``"on"``, ``"moderate"``, or ``"off"``.
        :param search_params:
            Additional keyword arguments forwarded to ``DDGS().text()`` (for example ``page`` or
            ``timelimit``). Values here override ``backend``, ``region``, ``safesearch``, and ``top_k``
            on conflict.
        """
        self.top_k = top_k
        self.backend = backend
        self.region = region
        self.safesearch = safesearch
        self.search_params = search_params or {}
        self._client: DDGS | None = None

    def warm_up(self) -> None:
        """
        Initialize the ddgs client.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._client is None:
            self._client = DDGS()

    def _search(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Run the blocking ddgs search and normalise the results.

        :param query:
            Search query.
        :returns:
            A dictionary with ``documents`` and ``links`` keys.
        """
        if self._client is None:
            self.warm_up()
        if self._client is None:
            msg = "DDGSWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params: dict[str, Any] = {
            "region": self.region,
            "safesearch": self.safesearch,
            "backend": self.backend,
            "max_results": self.top_k,
        }
        params.update(self.search_params)

        results = self._client.text(query, **params)
        documents, links = self._parse_response(results)

        logger.debug(
            "ddgs returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents, "links": links}

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Use ddgs to search the web.

        :param query:
            Search query.
        :returns:
            A dictionary with the following keys:
            - ``documents``: List of documents returned by the search backends.
            - ``links``: List of links returned by the search backends.
        """
        return self._search(query)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously use ddgs to search the web.

        ddgs has no native async API, so the blocking search runs in a worker thread. Same parameters
        and return values as :meth:`run`.

        :param query:
            Search query.
        :returns:
            A dictionary with ``documents`` and ``links`` keys.
        """
        return await asyncio.to_thread(self._search, query)

    @staticmethod
    def _parse_response(results: list[dict[str, Any]]) -> tuple[list[Document], list[str]]:
        """
        Convert ddgs text results into Haystack Documents and links.

        :param results:
            The list of result dictionaries returned by ``DDGS().text()``.
        :returns:
            A tuple of ``(documents, links)``.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            content = result.get("body", "")
            documents.append(Document(content=content, meta={"title": title, "url": url}))
            if url:
                links.append(url)

        return documents, links
