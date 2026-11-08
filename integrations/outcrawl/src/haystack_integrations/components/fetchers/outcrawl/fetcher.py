# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

OUTCRAWL_SCRAPE_URL = "https://api.outcrawl.ai/v1/scrape"


@component
class OutcrawlFetcher:
    """
    A component that uses Outcrawl to fetch web pages and return their content as Haystack Documents.

    Each URL is loaded in a browser by the Outcrawl Scrape API, so JavaScript-rendered pages work too,
    and the page content comes back as Markdown, which is well suited for LLMs.

    You need an Outcrawl API key from [outcrawl.ai](https://outcrawl.ai/console); the free plan includes
    credits every month.

    ### Usage example

    ```python
    from haystack_integrations.components.fetchers.outcrawl import OutcrawlFetcher
    from haystack.utils import Secret

    fetcher = OutcrawlFetcher(api_key=Secret.from_env_var("OUTCRAWL_API_KEY"))

    result = fetcher.run(urls=["https://docs.haystack.deepset.ai/docs/intro"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("OUTCRAWL_API_KEY"),
        params: dict[str, Any] | None = None,
        timeout: float = 65.0,
    ) -> None:
        """
        Initialize the OutcrawlFetcher.

        :param api_key:
            API key for Outcrawl. Defaults to the `OUTCRAWL_API_KEY` environment variable.
        :param params:
            Additional parameters for the scrape request, merged into the request body next to the URL.
            See the [Outcrawl API reference](https://docs.outcrawl.ai) for the available options.
            Defaults to `{"formats": ["markdown"]}`.
        :param timeout:
            Request timeout in seconds.
        """
        self.api_key = api_key
        self.params = params
        self.timeout = timeout
        self._params = {} if params is None else params.copy()
        self._params.setdefault("formats", ["markdown"])
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.resolve_value() or ''}",
            "Content-Type": "application/json",
        }

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

    @component.output_types(documents=list[Document])
    def run(self, urls: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Fetch the given URLs and return their content as Documents.

        URLs that fail to load are logged and skipped, so one bad page does not stop the others.

        :param urls:
            List of URLs to fetch.
        :param params:
            Optional per-run override of the scrape parameters.
            If provided, fully replaces the init-time `params`.
        :returns: A dictionary with:
            - `documents`: List of Documents, one for each URL that was fetched.
        """
        if self._client is None:
            self.warm_up()

        current_params = params if params is not None else self._params
        documents: list[Document] = []
        for url in urls:
            try:
                response = self._client.post(  # type: ignore[union-attr]
                    OUTCRAWL_SCRAPE_URL,
                    headers=self._build_headers(),
                    json={"url": url, **current_params},
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as error:
                logger.warning("Failed to fetch {url} with Outcrawl: {error}", url=url, error=error)
                continue
            document = self._document_from_response(url, payload)
            if document is not None:
                documents.append(document)

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(self, urls: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Asynchronously fetch the given URLs and return their content as Documents.

        URLs that fail to load are logged and skipped, so one bad page does not stop the others.

        :param urls:
            List of URLs to fetch.
        :param params:
            Optional per-run override of the scrape parameters.
            If provided, fully replaces the init-time `params`.
        :returns: A dictionary with:
            - `documents`: List of Documents, one for each URL that was fetched.
        """
        if self._async_client is None:
            await self.warm_up_async()

        current_params = params if params is not None else self._params
        documents: list[Document] = []
        for url in urls:
            try:
                response = await self._async_client.post(  # type: ignore[union-attr]
                    OUTCRAWL_SCRAPE_URL,
                    headers=self._build_headers(),
                    json={"url": url, **current_params},
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as error:
                logger.warning("Failed to fetch {url} with Outcrawl: {error}", url=url, error=error)
                continue
            document = self._document_from_response(url, payload)
            if document is not None:
                documents.append(document)

        return {"documents": documents}

    @staticmethod
    def _document_from_response(url: str, payload: Any) -> Document | None:
        """
        Convert an Outcrawl scrape response to a Haystack Document.

        :param url: URL that was requested.
        :param payload: Decoded JSON response of the Outcrawl Scrape API.
        :returns: A Document, or `None` if the response has no Markdown content.
        """
        markdown = payload.get("markdown") if isinstance(payload, dict) else None
        if not isinstance(markdown, str) or not markdown.strip():
            logger.warning("Outcrawl returned no content for {url}", url=url)
            return None

        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        meta: dict[str, Any] = {"url": payload.get("url") or url}
        if metadata.get("title"):
            meta["title"] = metadata["title"]
        if metadata.get("language"):
            meta["language"] = metadata["language"]
        if payload.get("statusCode") is not None:
            meta["status_code"] = payload["statusCode"]
        return Document(content=markdown, meta=meta)
