# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1/search"
_INTEGRATION_SLUG = "haystack"
_PACKAGE_NAME = "parallel-haystack"


def _attribution_header() -> str:
    try:
        version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{_INTEGRATION_SLUG}/{version}"


@component
class ParallelWebSearch:
    """
    A component that uses Parallel to search the web and return results as Haystack Documents.

    This component wraps the Parallel Search API, enabling web search queries that return
    LLM-optimized excerpts as structured documents with content and links.

    You need a Parallel API key from [parallel.ai](https://parallel.ai).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.parallel import ParallelWebSearch
    from haystack.utils import Secret

    websearch = ParallelWebSearch(
        api_key=Secret.from_env_var("PARALLEL_API_KEY"),
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
        api_key: Secret = Secret.from_env_var("PARALLEL_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the ParallelWebSearch component.

        :param api_key:
            API key for Parallel. Defaults to the `PARALLEL_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. Maps to the `advanced_settings.max_results` API parameter.
        :param search_params:
            Additional parameters passed to the Parallel Search API.
            See the [Parallel Search API reference](https://docs.parallel.ai/api-reference/search/search)
            for available options. Supported keys include: `objective` (natural-language search goal,
            defaults to the query), `mode` (`turbo`, `basic`, or `advanced`), `max_chars_total`,
            `session_id`, `client_model`, and `advanced_settings` (nested `source_policy` domain and
            date filters, `fetch_policy`, `excerpt_settings`, `location`, `max_results`).
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
            "x-api-key": self.api_key.resolve_value() or "",
            "Content-Type": "application/json",
            "x-parallel-integration": _attribution_header(),
        }

    def _build_body(self, query: str, search_params: dict[str, Any] | None) -> dict[str, Any]:
        params = (search_params if search_params is not None else self.search_params or {}).copy()
        advanced_settings = dict(params.pop("advanced_settings", None) or {})
        if "max_results" not in advanced_settings and self.top_k is not None:
            advanced_settings["max_results"] = self.top_k
        body: dict[str, Any] = {"search_queries": [query], "objective": query}
        body.update({k: v for k, v in params.items() if v is not None})
        if advanced_settings:
            body["advanced_settings"] = advanced_settings
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
        Search the web using Parallel and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result excerpts.
            - `links`: List of URLs from the search results.
        """
        if self._client is None:
            self.warm_up()

        response = self._client.post(  # type: ignore[union-attr]
            PARALLEL_SEARCH_URL,
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
        Asynchronously search the web using Parallel and return results as Documents.

        :param query: Search query string.
        :param search_params:
            Optional per-run override of search parameters.
            If provided, fully replaces the init-time `search_params`.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result excerpts.
            - `links`: List of URLs from the search results.
        """
        if self._async_client is None:
            self.warm_up()

        response = await self._async_client.post(  # type: ignore[union-attr]
            PARALLEL_SEARCH_URL,
            headers=self._build_headers(),
            json=self._build_body(query, search_params),
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a Parallel search response to Haystack Documents and links.

        :param response: Parallel search response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        for result in response.get("results", []):
            url = result.get("url", "")
            title = result.get("title") or ""
            excerpts = result.get("excerpts") or []
            content = " ... ".join(excerpts)
            meta: dict[str, Any] = {"title": title, "url": url}
            if excerpts:
                meta["excerpts"] = excerpts
            publish_date = result.get("publish_date")
            if publish_date is not None:
                meta["publish_date"] = publish_date
            documents.append(Document(content=content, meta=meta))
            if url:
                links.append(url)

        return {"documents": documents, "links": links}
