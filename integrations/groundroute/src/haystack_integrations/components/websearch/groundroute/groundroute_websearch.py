# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


GROUNDROUTE_BASE_URL = "https://api.groundroute.ai/v1/search"


class GroundRouteError(ComponentError):
    """Raised when a request to the GroundRoute API fails."""


@component
class GroundRouteWebSearch:
    """
    Uses [GroundRoute](https://groundroute.ai/) to search the web and return results as Documents.

    GroundRoute is a meta search layer: one API in front of six search engines (Serper, Brave, Exa,
    Tavily, Firecrawl, Perplexity). It routes each query to the cheapest engine that clears a quality
    bar and fails over to another engine if one is unavailable, so a pipeline keeps working without
    wiring up each engine separately. Pricing is gain-share: the caller keeps about half of any cache
    savings. You need a GroundRoute API key from [groundroute.ai](https://groundroute.ai/keys).

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.groundroute import GroundRouteWebSearch
    from haystack.utils import Secret

    websearch = GroundRouteWebSearch(
        api_key=Secret.from_env_var("GROUNDROUTE_API_KEY"),
        top_k=5,
    )
    result = websearch.run(query="What is a vector database?")
    documents = result["documents"]
    links = result["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("GROUNDROUTE_API_KEY"),
        top_k: int | None = 10,
        allowed_domains: list[str] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the GroundRouteWebSearch component.

        :param api_key:
            API key for GroundRoute. Defaults to the `GROUNDROUTE_API_KEY` environment variable.
        :param top_k:
            Maximum number of documents to return.
        :param allowed_domains:
            List of domains to restrict the search to.
        :param search_params:
            Additional parameters passed to the GroundRoute API, for example `mode`, `freshness`,
            `lang`, or `country`. See [groundroute.ai/docs](https://groundroute.ai/docs) for details.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}

        # Ensure that the API key is resolved.
        _ = self.api_key.resolve_value()

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            top_k=self.top_k,
            allowed_domains=self.allowed_domains,
            search_params=self.search_params,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroundRouteWebSearch":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, Any]:
        """
        Use GroundRoute to search the web.

        :param query: Search query.
        :returns: A dictionary with:
            - `documents`: List of Documents returned by the search engine.
            - `links`: List of URLs returned by the search engine.
        :raises GroundRouteError: If an error occurs while querying the GroundRoute API.
        :raises TimeoutError: If the request to the GroundRoute API times out.
        """
        payload, headers = self._prepare_request(query)
        try:
            response = httpx.post(GROUNDROUTE_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except httpx.ConnectTimeout as error:
            msg = f"Request to {self.__class__.__name__} timed out."
            raise TimeoutError(msg) from error
        except httpx.HTTPStatusError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            raise GroundRouteError(msg) from e
        except httpx.HTTPError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}"
            raise GroundRouteError(msg) from e

        documents, links = self._parse_response(response)

        logger.debug(
            "GroundRoute returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, Any]:
        """
        Asynchronously use GroundRoute to search the web.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param query: Search query.
        :returns: A dictionary with:
            - `documents`: List of Documents returned by the search engine.
            - `links`: List of URLs returned by the search engine.
        :raises GroundRouteError: If an error occurs while querying the GroundRoute API.
        :raises TimeoutError: If the request to the GroundRoute API times out.
        """
        payload, headers = self._prepare_request(query)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(GROUNDROUTE_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except httpx.ConnectTimeout as error:
            msg = f"Request to {self.__class__.__name__} timed out."
            raise TimeoutError(msg) from error
        except httpx.HTTPStatusError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            raise GroundRouteError(msg) from e
        except httpx.HTTPError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}"
            raise GroundRouteError(msg) from e

        documents, links = self._parse_response(response)

        logger.debug(
            "GroundRoute returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    def _prepare_request(self, query: str) -> tuple[dict[str, Any], dict[str, str]]:
        if (api_key := self.api_key.resolve_value()) is None:
            msg = "API key cannot be `None`."
            raise ValueError(msg)
        payload: dict[str, Any] = {"query": query, "max_results": self.top_k or 10, **self.search_params}
        if self.allowed_domains:
            payload["domains"] = self.allowed_domains
        headers = {"Authorization": f"Bearer {api_key}"}
        return payload, headers

    @staticmethod
    def _parse_response(response: httpx.Response) -> tuple[list[Document], list[str]]:
        json_result = response.json()
        results = json_result.get("results") or []

        documents: list[Document] = []

        # An answer-engine query returns a synthesized answer plus citations; surface it as a lead
        # document so it is available alongside the per-result documents.
        answer = json_result.get("answer")
        if answer:
            documents.append(
                Document(
                    content=answer,
                    meta={"source_engine": "groundroute", "citations": json_result.get("citations", [])},
                )
            )

        for r in results:
            content = r.get("content") or r.get("snippet") or ""
            meta = {"title": r.get("title", ""), "link": r.get("url", ""), "source_engine": r.get("source_engine", "")}
            if r.get("published_at"):
                meta["published_at"] = r["published_at"]
            documents.append(Document(content=content, meta=meta))

        links = [r.get("url", "") for r in results if r.get("url")]
        return documents, links
