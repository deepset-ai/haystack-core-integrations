# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
_INTEGRATION_SLUG = "haystack"
_PACKAGE_NAME = "perplexity-haystack"


def _attribution_header() -> str:
    try:
        version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{_INTEGRATION_SLUG}/{version}"


@component
class PerplexityWebSearch:
    """
    A component that uses Perplexity to search the web and return results as Haystack Documents.

    This component wraps the Perplexity Search API, enabling web search queries that return
    structured documents with content and links.

    You need a Perplexity API key from [perplexity.ai](https://www.perplexity.ai/).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.perplexity import PerplexityWebSearch
    from haystack.utils import Secret

    websearch = PerplexityWebSearch(
        api_key=Secret.from_env_var("PERPLEXITY_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the PerplexityWebSearch component.

        :param api_key:
            API key for Perplexity. Defaults to the `PERPLEXITY_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. Maps to the `max_results` API parameter (1-20).
        :param search_params:
            Additional parameters passed to the Perplexity Search API.
            See the [Perplexity Search API reference](https://docs.perplexity.ai/api-reference/search-post)
            for available options. Supported keys include: `max_tokens_per_page`, `country`,
            `search_recency_filter`, `search_domain_filter`, `search_language_filter`,
            `last_updated_after_filter`, `last_updated_before_filter`,
            `search_after_date_filter`, `search_before_date_filter`.
        :param timeout:
            Request timeout in seconds.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.resolve_value()}",
            "Content-Type": "application/json",
            "X-Pplx-Integration": _attribution_header(),
        }

    def _build_body(self, query: str, search_params: dict[str, Any] | None) -> dict[str, Any]:
        params = (search_params if search_params is not None else self.search_params or {}).copy()
        if "max_results" not in params and self.top_k is not None:
            params["max_results"] = self.top_k
        body: dict[str, Any] = {"query": query}
        body.update({k: v for k, v in params.items() if v is not None})
        return body

    def warm_up(self) -> None:
        """
        Initialize the sync and async HTTP clients.

        Called automatically on first use. Can be called explicitly to avoid cold-start latency.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document] | list[str]]:
        """
        Search the web using Perplexity and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._client is None:
            self.warm_up()

        response = self._client.post(  # type: ignore[union-attr]
            PERPLEXITY_SEARCH_URL,
            headers=self._build_headers(),
            json=self._build_body(query, search_params),
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously search the web using Perplexity and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        if self._async_client is None:
            self.warm_up()

        response = await self._async_client.post(  # type: ignore[union-attr]
            PERPLEXITY_SEARCH_URL,
            headers=self._build_headers(),
            json=self._build_body(query, search_params),
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a Perplexity search response to Haystack Documents and links.

        :param response: Perplexity search response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in response.get("results", []):
            url = result.get("url", "")
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            meta = {"title": title, "url": url}
            for optional_key in ("date", "last_updated"):
                value = result.get(optional_key)
                if value is not None:
                    meta[optional_key] = value
            documents.append(Document(content=snippet, meta=meta))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
