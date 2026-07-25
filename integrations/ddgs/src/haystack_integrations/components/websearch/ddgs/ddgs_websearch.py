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
            Comma-separated ddgs backends to query, or `"auto"` to let ddgs choose
            (for example `"duckduckgo, google, brave"`). See the ddgs docs for the full list.
        :param region:
            Region/locale for the search, for example `"us-en"`, `"de-de"`, or `"wt-wt"` (no region).
        :param safesearch:
            Safe-search level: `"on"`, `"moderate"`, or `"off"`.
        :param search_params:
            Additional keyword arguments forwarded to `DDGS().text()` (for example `page` or
            `timelimit`). Values here override `backend`, `region`, `safesearch`, and `top_k`
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

    def _search(
        self,
        query: str,
        top_k: int | None = None,
        backend: str | None = None,
        region: str | None = None,
        safesearch: str | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document] | list[str]]:
        """
        Run the blocking ddgs search and normalise the results.

        :param query:
            Search query.
        :returns:
            A dictionary with `documents` and `links` keys.
        """
        if self._client is None:
            self.warm_up()
        if self._client is None:
            msg = "DDGSWebSearch client failed to initialize."
            raise RuntimeError(msg)

        params: dict[str, Any] = {
            "region": region if region is not None else self.region,
            "safesearch": safesearch if safesearch is not None else self.safesearch,
            "backend": backend if backend is not None else self.backend,
            "max_results": top_k if top_k is not None else self.top_k,
        }
        params.update(search_params if search_params is not None else self.search_params)

        results = self._client.text(query, **params)
        documents, links = self._parse_response(results)

        logger.debug(
            "ddgs returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents, "links": links}

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        *,
        backend: str | None = None,
        region: str | None = None,
        safesearch: str | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document] | list[str]]:
        """
        Use ddgs to search the web.

        :param query:
            Search query.
        :param top_k:
            Optional per-run override of the maximum number of results. If not provided, the
            init-time `top_k` is used.
        :param backend:
            Optional per-run override of the ddgs backends. If not provided, the init-time
            `backend` is used.
        :param region:
            Optional per-run override of the region/locale. If not provided, the init-time
            `region` is used.
        :param safesearch:
            Optional per-run override of the safe-search level. If not provided, the init-time
            `safesearch` is used.
        :param search_params:
            Optional per-run override of the extra `DDGS().text()` arguments. If provided, fully
            replaces the init-time `search_params`.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of documents returned by the search backends.
            - `links`: List of links returned by the search backends.
        """
        return self._search(query, top_k, backend, region, safesearch, search_params)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
        *,
        backend: str | None = None,
        region: str | None = None,
        safesearch: str | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously use ddgs to search the web.

        ddgs has no native async API, so the blocking search runs in a worker thread. Same parameters
        and return values as :meth:`run`.

        :param query:
            Search query.
        :param top_k:
            Optional per-run override of the maximum number of results. If not provided, the
            init-time `top_k` is used.
        :param backend:
            Optional per-run override of the ddgs backends. If not provided, the init-time
            `backend` is used.
        :param region:
            Optional per-run override of the region/locale. If not provided, the init-time
            `region` is used.
        :param safesearch:
            Optional per-run override of the safe-search level. If not provided, the init-time
            `safesearch` is used.
        :param search_params:
            Optional per-run override of the extra `DDGS().text()` arguments. If provided, fully
            replaces the init-time `search_params`.
        :returns:
            A dictionary with `documents` and `links` keys.
        """
        return await asyncio.to_thread(self._search, query, top_k, backend, region, safesearch, search_params)

    @staticmethod
    def _parse_response(results: list[dict[str, Any]]) -> tuple[list[Document], list[str]]:
        """
        Convert ddgs text results into Haystack Documents and links.

        :param results:
            The list of result dictionaries returned by `DDGS().text()`.
        :returns:
            A tuple of `(documents, links)`.
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
