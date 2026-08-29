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
class ContextCrawler:
    """
    Crawl one or more websites with Context.dev and return each page as a Haystack Document.

    Crawls are bounded to one page by default to prevent accidental credit consumption. Increase `maxPages` through
    `crawl_params` when a pipeline needs a larger site ingest. Create an API key in the
    [Context.dev dashboard](https://www.context.dev/dashboard/api-keys) and set it as `CONTEXT_API_KEY`.

    ### Usage example

    ```python
    from haystack_integrations.components.fetchers.context import ContextCrawler

    crawler = ContextCrawler(crawl_params={"maxPages": 10, "maxDepth": 2})
    result = crawler.run(urls=["https://docs.haystack.deepset.ai"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(API_KEY_ENV_VAR),
        *,
        crawl_params: dict[str, Any] | None = None,
        api_url: str = DEFAULT_API_URL,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Context.dev crawler component.

        :param api_key: Context.dev API key. Defaults to the `CONTEXT_API_KEY` environment variable.
        :param crawl_params: Additional parameters passed to the Context.dev Crawl API.
        :param api_url: Base URL for the Context.dev API.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum number of retry attempts on transient failures.
        """
        self.api_key = api_key
        self.crawl_params = crawl_params
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document])
    def run(
        self,
        urls: list[str],
        crawl_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Crawl the given URLs and return successful pages as Documents.

        :param urls: Starting URLs to crawl.
        :param crawl_params: Optional per-run replacement for init-time crawl parameters.
        :returns: A dictionary containing `documents`.
        """
        documents = [document for url in urls for document in self._crawl_url(url, crawl_params)]
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        urls: list[str],
        crawl_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously crawl the given URLs and return successful pages as Documents.

        :param urls: Starting URLs to crawl concurrently.
        :param crawl_params: Optional per-run replacement for init-time crawl parameters.
        :returns: A dictionary containing `documents`.
        """
        crawls = await asyncio.gather(*(self._crawl_url_async(url, crawl_params) for url in urls))
        documents = [document for crawl in crawls for document in crawl]
        return {"documents": documents}

    def _crawl_url(self, url: str, crawl_params: dict[str, Any] | None) -> list[Document]:
        response = request_context(
            api_key=self.api_key,
            api_url=self.api_url,
            method="POST",
            path="/web/crawl",
            json=self._request_body(url, crawl_params),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._documents_from_response(response)

    async def _crawl_url_async(self, url: str, crawl_params: dict[str, Any] | None) -> list[Document]:
        response = await request_context_async(
            api_key=self.api_key,
            api_url=self.api_url,
            method="POST",
            path="/web/crawl",
            json=self._request_body(url, crawl_params),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._documents_from_response(response)

    def _request_body(self, url: str, crawl_params: dict[str, Any] | None) -> dict[str, Any]:
        params = (crawl_params if crawl_params is not None else self.crawl_params or {}).copy()
        defaults = {
            "maxPages": 1,
            "useMainContentOnly": True,
            "includeLinks": True,
            "includeImages": False,
        }
        return {**defaults, **params, "url": url}

    @staticmethod
    def _documents_from_response(response: dict[str, Any]) -> list[Document]:
        results = response.get("results", [])
        if not isinstance(results, list):
            return []

        documents: list[Document] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict) or not metadata.get("success", False):
                continue
            content = result.get("markdown", "")
            documents.append(Document(content=content if isinstance(content, str) else "", meta=metadata.copy()))
        return documents
