# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret

from firecrawl import AsyncFirecrawl, Firecrawl  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@component
class FirecrawlCrawler:
    """
    A component that uses Firecrawl to crawl one or more URLs and return the content as Haystack Documents.

    Crawling starts from each given URL and follows links to discover subpages, up to a configurable limit.
    This is useful for ingesting entire websites or documentation sites, not just single pages.

    Firecrawl is a service that crawls websites and returns content in a structured format (e.g. Markdown)
    suitable for LLMs. You need a Firecrawl API key from [firecrawl.dev](https://firecrawl.dev).

    ### Usage example

    ```python
    from haystack_integrations.components.fetchers.firecrawl import FirecrawlFetcher

    fetcher = FirecrawlFetcher(
        api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
        params={"limit": 5},
    )
    fetcher.warm_up()

    result = fetcher.run(urls=["https://docs.haystack.deepset.ai/docs/intro"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("FIRECRAWL_API_KEY"),
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the FirecrawlFetcher.

        :param api_key:
            API key for Firecrawl.
            Defaults to the `FIRECRAWL_API_KEY` environment variable.
        :param params:
            Parameters for the crawl request. See the
            [Firecrawl API reference](https://docs.firecrawl.dev/api-reference/endpoint/crawl-post)
            for available parameters.
            Defaults to `{"limit": 1, "scrape_options": {"formats": ["markdown"]}}`.
            Without a limit, Firecrawl may crawl all subpages and consume credits quickly.
        """
        self.api_key = api_key
        self.params = params
        self._params = {} if params is None else params.copy()
        self._params.setdefault("limit", 1)
        self._params.setdefault("scrape_options", {"formats": ["markdown"]})
        self._firecrawl_client: Firecrawl | None = None
        self._async_firecrawl_client: AsyncFirecrawl | None = None

    @component.output_types(documents=list[Document])
    def run(
        self,
        urls: list[str],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Crawls the given URLs and returns the extracted content as Documents.

        :param urls:
            List of URLs to crawl.
        :param params:
            Optional override of crawl parameters for this run.
            If provided, fully replaces the init-time params.
        :returns: A dictionary with the following keys:
            - `documents`: List of documents, one for each URL crawled.
        """
        if self._firecrawl_client is None:
            self.warm_up()

        current_params = params if params is not None else self._params
        documents: list[Document] = []
        for url in urls:
            docs = self._crawl_url(url=url, params=current_params)
            documents.extend(docs)

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        urls: list[str],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously crawls the given URLs and returns the extracted content as Documents.

        :param urls:
            List of URLs to crawl.
        :param params:
            Optional override of crawl parameters for this run.
            If provided, fully replaces the init-time params.
        :returns: A dictionary with the following keys:
            - `documents`: List of documents, one for each URL crawled.
        """
        if self._async_firecrawl_client is None:
            self.warm_up()

        current_params = params if params is not None else self._params
        documents: list[Document] = []
        for url in urls:
            docs = await self._crawl_url_async(url=url, params=current_params)
            documents.extend(docs)

        return {"documents": documents}

    def warm_up(self) -> None:
        """
        Warm up the Firecrawl client by initializing the clients.
        This is useful to avoid cold start delays when crawling many URLs.
        """
        if self._firecrawl_client is None:
            self._firecrawl_client = Firecrawl(api_key=self.api_key.resolve_value())
        if self._async_firecrawl_client is None:
            self._async_firecrawl_client = AsyncFirecrawl(api_key=self.api_key.resolve_value())

    def _crawl_url(self, url: str, params: dict[str, Any]) -> list[Document]:
        """
        Crawl a single URL and return Documents.

        :param url: URL to crawl.
        :param params: Crawl request parameters.
        :return: List of Documents from the crawl result.
        """
        try:
            # Ignoring type because the firecrawl client is initialized as None and only set during warm_up
            crawl_response = self._firecrawl_client.crawl(  # type: ignore[union-attr]
                url=url,
                **params,
            )
        except Exception as error:
            logger.exception(f"Failed to crawl website {url}: {error}")
            return []

        return self._documents_from_crawl_response(url=url, crawl_response=crawl_response)

    async def _crawl_url_async(self, url: str, params: dict[str, Any]) -> list[Document]:
        """
        Asynchronously crawl a single URL and return Documents.

        :param url: URL to crawl.
        :param params: Crawl request parameters.
        :return: List of Documents from the crawl result.
        """
        try:
            # Ignoring type because the firecrawl client is initialized as None and only set during warm_up
            crawl_response = await self._async_firecrawl_client.crawl(  # type: ignore[union-attr]
                url=url,
                **params,
            )
        except Exception as error:
            logger.exception(f"Failed to crawl website {url}: {error}")
            return []

        return self._documents_from_crawl_response(url=url, crawl_response=crawl_response)

    def _documents_from_crawl_response(self, url: str, crawl_response: Any) -> list[Document]:
        """
        Convert a Firecrawl crawl response to Haystack Documents.

        :param url: URL that was crawled.
        :param crawl_response: Firecrawl crawl response.
        :return: List of documents built from crawled pages.
        """
        if crawl_response.status != "completed":
            logger.exception(f"Failed to crawl website {url}: {crawl_response.status}")

        documents: list[Document] = []
        for page in crawl_response.data:
            metadata = page.metadata_dict
            doc = Document(
                content=page.markdown,
                meta={
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    **metadata,
                },
            )
            documents.append(doc)

        return documents
