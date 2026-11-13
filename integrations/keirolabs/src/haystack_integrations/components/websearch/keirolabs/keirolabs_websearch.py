# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import ComponentError, Document, component
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry

KEIROLABS_SEARCH_URL = "https://api.keirolabs.cloud/api/v2/search/lite"


class KeirolabsError(ComponentError):
    """An error occurred while querying the Keirolabs search API."""


@component
class KeirolabsWebSearch:
    """
    A component that uses the [Keirolabs](https://keirolabs.cloud) search API to search the web.

    Each result becomes a Haystack Document whose content is the result snippet, with the title and
    URL in its metadata. Keirolabs charges a flat price per search whatever the result count.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.keirolabs import KeirolabsWebSearch

    websearch = KeirolabsWebSearch(top_k=5)  # reads KEIROLABS_API_KEY
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("KEIROLABS_API_KEY"),
        top_k: int | None = 10,
        allowed_domains: list[str] | None = None,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the KeirolabsWebSearch component.

        :param api_key:
            Keirolabs API key. Defaults to the `KEIROLABS_API_KEY` environment variable.
            Get one at https://platform.keirolabs.cloud.
        :param top_k:
            Maximum number of results to return. Maps to `maxResults` in the Keirolabs API,
            which returns at most about 15 results per search.
        :param allowed_domains:
            Restrict results to these domains, added to the query as `site:` clauses.
        :param timeout:
            Timeout in seconds for the HTTP request. Defaults to 10.
        :param max_retries:
            Maximum number of retry attempts on transient failures. Defaults to 3.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Search the web using Keirolabs and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result snippets.
            - `links`: List of URLs from the search results.
        :raises KeirolabsError: If the Keirolabs request fails.
        """
        try:
            response = request_with_retry(
                attempts=self.max_retries,
                method="POST",
                url=KEIROLABS_SEARCH_URL,
                json=self._build_payload(query, top_k),
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise KeirolabsError(self._error_message(error)) from error
        return self._parse_response(response.json(), self._limit(top_k))

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Asynchronously search the web using Keirolabs and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result snippets.
            - `links`: List of URLs from the search results.
        :raises KeirolabsError: If the Keirolabs request fails.
        """
        try:
            response = await async_request_with_retry(
                attempts=self.max_retries,
                method="POST",
                url=KEIROLABS_SEARCH_URL,
                json=self._build_payload(query, top_k),
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise KeirolabsError(self._error_message(error)) from error
        return self._parse_response(response.json(), self._limit(top_k))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.resolve_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _limit(self, top_k: int | None) -> int | None:
        return top_k if top_k is not None else self.top_k

    def _build_payload(self, query: str, top_k: int | None) -> dict[str, Any]:
        if self.allowed_domains:
            sites = " OR ".join(f"site:{domain}" for domain in self.allowed_domains)
            query = f"({query}) ({sites})"
        payload: dict[str, Any] = {"query": query}
        limit = self._limit(top_k)
        if limit is not None:
            payload["maxResults"] = limit
        return payload

    @staticmethod
    def _error_message(error: httpx.HTTPError) -> str:
        message = f"An error occurred while querying the Keirolabs search API. Error: {error}"
        if isinstance(error, httpx.HTTPStatusError):
            message = f"{message}, Response: {error.response.text}"
        return message

    @staticmethod
    def _parse_response(response: dict[str, Any], limit: int | None) -> dict[str, Any]:
        documents: list[Document] = []
        links: list[str] = []
        for result in response.get("results") or []:
            url = result.get("url")
            if not url:
                continue
            meta = {"title": result.get("title") or "", "url": url}
            if result.get("domain"):
                meta["domain"] = result["domain"]
            documents.append(Document(content=result.get("snippet") or "", meta=meta))
            links.append(url)
        if limit is not None:
            documents, links = documents[:limit], links[:limit]
        return {"documents": documents, "links": links}
