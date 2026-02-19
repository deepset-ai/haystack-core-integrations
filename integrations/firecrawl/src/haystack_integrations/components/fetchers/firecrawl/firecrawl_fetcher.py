# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Any

from firecrawl import AsyncFirecrawl, Firecrawl
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class FirecrawlFetcher:
    """
    A component that uses Firecrawl to crawl one or more URLs and return the content as Haystack Documents.

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

    def warm_up(self) -> None:
        """Initialize the Firecrawl client. Call this before the first `run` (or rely on lazy init in `run`)."""
        if self._firecrawl_client is None:
            self._firecrawl_client = Firecrawl(api_key=self.api_key.resolve_value())
        if self._async_firecrawl_client is None:
            self._async_firecrawl_client = AsyncFirecrawl(api_key=self.api_key.resolve_value())

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes a FirecrawlFetcher instance to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            params=self.params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FirecrawlFetcher":
        """
        Deserializes a FirecrawlFetcher instance from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized FirecrawlFetcher instance.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        return default_from_dict(cls, data)

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
        :returns: A dictionary with key `documents` containing a list of Haystack `Document` instances.
        """
        if self._firecrawl_client is None:
            self.warm_up()

        current_params = dict(self._params, **(params or {}))
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
        :returns: A dictionary with key `documents` containing a list of Haystack `Document` instances.
        """
        if self._async_firecrawl_client is None:
            self.warm_up()

        current_params = dict(self._params, **(params or {}))
        documents: list[Document] = []
        for url in urls:
            docs = await self._crawl_url_async(url=url, params=current_params)
            documents.extend(docs)

        return {"documents": documents}

    def _crawl_url(self, url: str, params: dict[str, Any]) -> list[Document]:
        """
        Crawl a single URL and return Documents.

        :param url: URL to crawl.
        :param params: Crawl request parameters.
        :return: List of Documents from the crawl result.
        """

        try:
            crawl_response = self._firecrawl_client.crawl(
                url=url,
                **params,
            )
        except Exception as error:
            logger.error("Failed to crawl website %s: %s", url, error)
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
            crawl_response = await self._async_firecrawl_client.crawl(
                url=url,
                **params,
            )
        except Exception as error:
            logger.error("Failed to crawl website %s: %s", url, error)
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
            logger.error("Failed to crawl website %s: %s", url, crawl_response.status)

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
