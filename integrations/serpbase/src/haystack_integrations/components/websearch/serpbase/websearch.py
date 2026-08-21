# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from urllib.parse import urlparse

import httpx
from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


SERPBASE_BASE_URL = "https://api.serpbase.dev/google/search"


class SerpBaseError(ComponentError): ...


@component
class SerpBaseWebSearch:
    """
    Uses [SerpBase](https://serpbase.dev/) to search the web for relevant documents.

    See the [SerpBase documentation](https://serpbase.dev/docs) for more details.

    Usage example:
    ```python
    from haystack.utils import Secret

    from haystack_integrations.components.websearch.serpbase import SerpBaseWebSearch

    serpbase_api = Secret.from_env_var("SERPBASE_API_KEY")

    websearch = SerpBaseWebSearch(top_k=10, api_key=serpbase_api)
    results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

    assert results["documents"]
    assert results["links"]

    # Example with domain filtering - exclude subdomains
    websearch_filtered = SerpBaseWebSearch(
        top_k=10,
        allowed_domains=["example.com"],
        exclude_subdomains=True,  # Only results from example.com, not blog.example.com
        api_key=serpbase_api,
    )
    results_filtered = websearch_filtered.run(query="search query")
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SERPBASE_API_KEY"),
        top_k: int | None = 10,
        allowed_domains: list[str] | None = None,
        search_params: dict[str, Any] | None = None,
        *,
        exclude_subdomains: bool = False,
    ) -> None:
        """
        Initialize the SerpBaseWebSearch component.

        :param api_key: API key for the SerpBase API.
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param exclude_subdomains: Whether to exclude subdomains when filtering by allowed_domains.
            If True, only results from the exact domains in allowed_domains will be returned.
            If False, results from subdomains will also be included. Defaults to False.
        :param search_params: Additional parameters passed to the SerpBase API.
            For example, you can set 'num' to 20 to increase the number of search results.
            See the [SerpBase documentation](https://serpbase.dev/docs) for more details.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.exclude_subdomains = exclude_subdomains
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
            exclude_subdomains=self.exclude_subdomains,
            search_params=self.search_params,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerpBaseWebSearch":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _is_domain_allowed(self, url: str) -> bool:
        """
        Check if a URL's domain is allowed based on allowed_domains and exclude_subdomains settings.

        :param url: The URL to check.
        :returns: True if the domain is allowed, False otherwise.
        """
        if not self.allowed_domains:
            return True

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            for allowed_domain in self.allowed_domains:
                allowed_domain_lower = allowed_domain.lower()

                if self.exclude_subdomains:
                    # Exact domain match only
                    if domain == allowed_domain_lower:
                        return True
                # Allow subdomains (current behavior)
                elif domain == allowed_domain_lower or domain.endswith("." + allowed_domain_lower):
                    return True

            return False
        except Exception:
            # If URL parsing fails, allow the result to be safe
            return True

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Use [SerpBase](https://serpbase.dev/) to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises SerpBaseError: If an error occurs while querying the SerpBase API.
        :raises TimeoutError: If the request to the SerpBase API times out.
        """
        payload, headers = self._prepare_request(query)
        try:
            response = httpx.post(SERPBASE_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except httpx.TimeoutException as error:
            msg = f"Request to {self.__class__.__name__} timed out."
            raise TimeoutError(msg) from error

        except httpx.HTTPStatusError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            raise SerpBaseError(msg) from e

        except httpx.HTTPError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}"
            raise SerpBaseError(msg) from e

        documents, links = self._parse_response(response)

        logger.debug(
            "SerpBase returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously uses [SerpBase](https://serpbase.dev/) to search the web.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises SerpBaseError: If an error occurs while querying the SerpBase API.
        :raises TimeoutError: If the request to the SerpBase API times out.
        """
        payload, headers = self._prepare_request(query)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(SERPBASE_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except httpx.TimeoutException as error:
            msg = f"Request to {self.__class__.__name__} timed out."
            raise TimeoutError(msg) from error

        except httpx.HTTPStatusError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            raise SerpBaseError(msg) from e

        except httpx.HTTPError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}"
            raise SerpBaseError(msg) from e

        documents, links = self._parse_response(response)

        logger.debug(
            "SerpBase returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    def _prepare_request(self, query: str) -> tuple[dict[str, Any], dict[str, str]]:
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""
        payload = {"q": query_prepend + query, "hl": "en", "gl": "us", "device": "default", **self.search_params}
        if (api_key := self.api_key.resolve_value()) is None:
            msg = "API key cannot be `None`."
            raise ValueError(msg)
        headers = {"X-API-Key": api_key}
        return payload, headers

    def _parse_response(self, response: httpx.Response) -> tuple[list[Document], list[str]]:
        # If we reached this point, it means the request was successful and we can proceed
        json_result = response.json()

        # Endpoint data can be nested under a "result" key depending on the SerpBase API version
        result = json_result.get("result") if isinstance(json_result.get("result"), dict) else json_result

        # we get the snippet from the json result and put it in the content field of the document
        organic = [
            Document(meta={k: v for k, v in d.items() if k != "snippet"}, content=d.get("snippet"))
            for d in result.get("organic", [])
            if self._is_domain_allowed(d.get("link", ""))
        ]

        # featured snippet is what search engine shows as a direct answer to the query
        featured_snippet = []
        if "featured_snippet" in result and isinstance(result["featured_snippet"], dict):
            snippet_dict = result["featured_snippet"]
            snippet_content = None
            for key in ["snippet", "answer", "title"]:
                if key in snippet_dict:
                    snippet_content = snippet_dict[key]
                    break
            if snippet_content and self._is_domain_allowed(snippet_dict.get("link", "")):
                featured_snippet = [
                    Document(
                        content=snippet_content,
                        meta={
                            "title": snippet_dict.get("title", ""),
                            "link": snippet_dict.get("link", ""),
                            "featured": True,
                        },
                    )
                ]

        # these are related questions that the search engine shows
        people_also_ask = []
        for result_dict in result.get("people_also_ask", []):
            link = result_dict.get("link")
            if self._is_domain_allowed(link or ""):
                title = result_dict.get("question", "")
                people_also_ask.append(
                    Document(
                        content=result_dict.get("answer") or title,
                        meta={"title": title, "link": link},
                    )
                )

        documents = featured_snippet + organic + people_also_ask

        links = [
            result_dict["link"]
            for result_dict in result.get("organic", [])
            if self._is_domain_allowed(result_dict.get("link", ""))
        ]
        return documents, links
