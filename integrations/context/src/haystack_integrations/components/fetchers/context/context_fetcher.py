# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from haystack import Document, component
from haystack.utils import Secret

from haystack_integrations.context._client import (
    API_KEY_ENV_VAR,
    DEFAULT_API_URL,
    request_context,
    request_context_async,
)


@component
class ContextFetcher:
    """
    Fetch known URLs as clean Markdown with Context.dev.

    The component converts each URL into a Haystack Document and preserves page metadata. Create an API key in the
    [Context.dev dashboard](https://www.context.dev/dashboard/api-keys) and set it as `CONTEXT_API_KEY`.

    ### Usage example

    ```python
    from haystack_integrations.components.fetchers.context import ContextFetcher

    fetcher = ContextFetcher()
    result = fetcher.run(urls=["https://haystack.deepset.ai"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(API_KEY_ENV_VAR),
        *,
        scrape_params: dict[str, Any] | None = None,
        api_url: str = DEFAULT_API_URL,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Context.dev fetcher component.

        :param api_key: Context.dev API key. Defaults to the `CONTEXT_API_KEY` environment variable.
        :param scrape_params: Additional query parameters passed to the Context.dev Markdown Scrape API.
        :param api_url: Base URL for the Context.dev API.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum number of retry attempts on transient failures.
        """
        self.api_key = api_key
        self.scrape_params = scrape_params
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document])
    def run(
        self,
        urls: list[str],
        scrape_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch the given URLs and return their Markdown as Documents.

        :param urls: URLs to fetch.
        :param scrape_params: Optional per-run replacement for init-time scrape parameters.
        :returns: A dictionary containing `documents`.
        """
        documents = [self._fetch_url(url, scrape_params) for url in urls]
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        urls: list[str],
        scrape_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously fetch the given URLs and return their Markdown as Documents.

        :param urls: URLs to fetch concurrently.
        :param scrape_params: Optional per-run replacement for init-time scrape parameters.
        :returns: A dictionary containing `documents`.
        """
        documents = await asyncio.gather(*(self._fetch_url_async(url, scrape_params) for url in urls))
        return {"documents": list(documents)}

    def _fetch_url(self, url: str, scrape_params: dict[str, Any] | None) -> Document:
        response = request_context(
            api_key=self.api_key,
            api_url=self.api_url,
            method="GET",
            path="/web/scrape/markdown",
            params=self._request_params(url, scrape_params),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._document_from_response(response, url)

    async def _fetch_url_async(self, url: str, scrape_params: dict[str, Any] | None) -> Document:
        response = await request_context_async(
            api_key=self.api_key,
            api_url=self.api_url,
            method="GET",
            path="/web/scrape/markdown",
            params=self._request_params(url, scrape_params),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._document_from_response(response, url)

    def _request_params(self, url: str, scrape_params: dict[str, Any] | None) -> dict[str, Any]:
        params = (scrape_params if scrape_params is not None else self.scrape_params or {}).copy()
        defaults = {
            "useMainContentOnly": True,
            "includeLinks": True,
            "includeImages": False,
        }
        return {**defaults, **params, "url": url}

    @staticmethod
    def _document_from_response(response: dict[str, Any], requested_url: str) -> Document:
        metadata = response.get("metadata", {})
        meta = metadata.copy() if isinstance(metadata, dict) else {}
        meta.update(
            {
                "url": response.get("url", requested_url),
                "content_length": response.get("contentLength"),
            }
        )
        content = response.get("markdown", "")
        return Document(content=content if isinstance(content, str) else "", meta=meta)
