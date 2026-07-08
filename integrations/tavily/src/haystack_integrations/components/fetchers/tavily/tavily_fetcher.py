# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import Document, component, logging
from haystack.utils import Secret

from tavily import AsyncTavilyClient, TavilyClient

logger = logging.getLogger(__name__)


@component
class TavilyFetcher:
    """
    A component that uses the Tavily Extract API to fetch and extract content from URLs as Haystack Documents.

    This component wraps the Tavily Extract API, which retrieves and parses web page content from
    one or more specified URLs. Unlike web search, it fetches content directly from the given URLs
    rather than discovering them via a query.

    Tavily is an AI-powered search and extraction API optimized for LLM applications. You need a Tavily
    API key from [tavily.com](https://tavily.com).

    ### Usage example

    ```python
    from haystack_integrations.components.fetchers.tavily import TavilyFetcher
    from haystack.utils import Secret

    fetcher = TavilyFetcher(
        api_key=Secret.from_env_var("TAVILY_API_KEY"),
        extract_depth="basic",
    )
    result = fetcher.run(urls=["https://haystack.deepset.ai"])
    documents = result["documents"]
    meta = result["meta"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TAVILY_API_KEY"),
        extract_depth: Literal["basic", "advanced"] = "basic",
        include_images: bool = False,
        extract_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the TavilyFetcher component.

        :param api_key:
            API key for Tavily. Defaults to the `TAVILY_API_KEY` environment variable.
        :param extract_depth:
            Extraction depth: `"basic"` (fast, lower cost) or `"advanced"` (more data including
            tables, higher latency and cost). Defaults to `"basic"`.
        :param include_images:
            If `True`, extracted image URLs are included in each Document's metadata under
            the `"images"` key. Defaults to `False`.
        :param extract_params:
            Additional parameters passed to the Tavily Extract API, such as `format`,
            `include_favicon`, `query`, or `chunks_per_source`.
            See the [Tavily Extract API reference](https://docs.tavily.com/documentation/api-reference/endpoint/extract)
            for available options.
        """
        self.api_key = api_key
        self.extract_depth = extract_depth
        self.include_images = include_images
        self.extract_params = extract_params
        self._tavily_client: TavilyClient | None = None
        self._async_tavily_client: AsyncTavilyClient | None = None

    def warm_up(self) -> None:
        """
        Initialize the Tavily sync and async clients.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._tavily_client is None:
            self._tavily_client = TavilyClient(api_key=self.api_key.resolve_value())
        if self._async_tavily_client is None:
            self._async_tavily_client = AsyncTavilyClient(api_key=self.api_key.resolve_value())

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(
        self,
        urls: list[str],
        extract_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch and extract content from the given URLs using the Tavily Extract API.

        :param urls:
            List of URLs to extract content from. Maximum 20 URLs per request.
        :param extract_params:
            Optional per-run override of extract parameters.
            If provided, fully replaces the init-time `extract_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing extracted page content.
              Each Document's `meta` includes `"url"` and, if `include_images` is True, `"images"`.
            - `meta`: Request-level metadata containing `"response_time"`, `"usage"`,
              `"request_id"`, and `"failed_results"` for URLs that could not be processed.
        """
        if self._tavily_client is None:
            self.warm_up()
        if self._tavily_client is None:
            msg = "TavilyFetcher client failed to initialize."
            raise RuntimeError(msg)

        params = (extract_params if extract_params is not None else self.extract_params or {}).copy()
        response = self._tavily_client.extract(
            urls=urls,
            extract_depth=self.extract_depth,
            include_images=self.include_images,
            **params,
        )
        return self._parse_response(response)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(
        self,
        urls: list[str],
        extract_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously fetch and extract content from the given URLs using the Tavily Extract API.

        :param urls:
            List of URLs to extract content from. Maximum 20 URLs per request.
        :param extract_params:
            Optional per-run override of extract parameters.
            If provided, fully replaces the init-time `extract_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing extracted page content.
              Each Document's `meta` includes `"url"` and, if `include_images` is True, `"images"`.
            - `meta`: Request-level metadata containing `"response_time"`, `"usage"`,
              `"request_id"`, and `"failed_results"` for URLs that could not be processed.
        """
        if self._async_tavily_client is None:
            self.warm_up()
        if self._async_tavily_client is None:
            msg = "TavilyFetcher async client failed to initialize."
            raise RuntimeError(msg)

        params = (extract_params if extract_params is not None else self.extract_params or {}).copy()
        response = await self._async_tavily_client.extract(
            urls=urls,
            extract_depth=self.extract_depth,
            include_images=self.include_images,
            **params,
        )
        return self._parse_response(response)

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a Tavily Extract response into Haystack Documents and request metadata.

        :param response: Raw Tavily Extract API response dictionary.
        :returns: Dictionary with `documents` and `meta` keys.
        """
        documents: list[Document] = []

        for result in response.get("results", []):
            url = result.get("url", "")
            content = result.get("raw_content", "")
            doc_meta: dict[str, Any] = {"url": url}
            if self.include_images:
                doc_meta["images"] = result.get("images", [])
            documents.append(Document(content=content, meta=doc_meta))

        meta: dict[str, Any] = {
            "response_time": response.get("response_time"),
            "usage": response.get("usage"),
            "request_id": response.get("request_id"),
            "failed_results": response.get("failed_results", []),
        }

        return {"documents": documents, "meta": meta}
