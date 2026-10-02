# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


SEARCHAPI_BASE_URL = "https://www.searchapi.io/api/v1/search"


class SearchApiError(ComponentError): ...


@component
class SearchApiWebSearch:
    """
    Uses [SearchApi](https://www.searchapi.io/) to search the web for relevant documents.

    Usage example:
    ```python
    from haystack.utils import Secret

    from haystack_integrations.components.websearch.searchapi import SearchApiWebSearch

    websearch = SearchApiWebSearch(top_k=10, api_key=Secret.from_env_var("SEARCHAPI_API_KEY"))
    results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

    assert results["documents"]
    assert results["links"]
    ```

    SearchApi's AI answer engines are supported through `search_params`. With `{"engine": "perplexity"}`,
    `{"engine": "google_ai_mode"}`, `{"engine": "chatgpt"}`, `{"engine": "gemini"}` or `{"engine": "bing_copilot"}`
    the first document holds the synthesized answer as Markdown and the following documents and `links` hold the
    reference links it cites.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SEARCHAPI_API_KEY"),
        top_k: int | None = 10,
        allowed_domains: list[str] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the SearchApiWebSearch component.

        :param api_key: API key for the SearchApi API
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param search_params: Additional parameters passed to the SearchApi API.
            For example, you can set 'num' to 100 to increase the number of search results.
            See the [SearchApi website](https://www.searchapi.io/) for more details.

            The default search engine is Google, however, users can change it by setting the `engine`
            parameter in the `search_params`.
        """

        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}
        if "engine" not in self.search_params:
            self.search_params["engine"] = "google"

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
    def from_dict(cls, data: dict[str, Any]) -> "SearchApiWebSearch":
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
    def run(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Uses [SearchApi](https://www.searchapi.io/) to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises TimeoutError: If the request to the SearchApi API times out.
        :raises SearchApiError: If an error occurs while querying the SearchApi API.
        """
        payload, headers = self._prepare_request(query)
        try:
            response = httpx.get(SEARCHAPI_BASE_URL, headers=headers, params=payload, timeout=90)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except httpx.ConnectTimeout as error:
            msg = f"Request to {self.__class__.__name__} timed out."
            raise TimeoutError(msg) from error

        except httpx.HTTPStatusError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            raise SearchApiError(msg) from e

        except httpx.HTTPError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}"
            raise SearchApiError(msg) from e

        documents, links = self._parse_response(response)

        logger.debug(
            "SearchApi returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously uses [SearchApi](https://www.searchapi.io/) to search the web.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises TimeoutError: If the request to the SearchApi API times out.
        :raises SearchApiError: If an error occurs while querying the SearchApi API.
        """
        payload, headers = self._prepare_request(query)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(SEARCHAPI_BASE_URL, headers=headers, params=payload, timeout=90)
                response.raise_for_status()  # Will raise an HTTPError for bad responses
        except httpx.ConnectTimeout as error:
            msg = f"Request to {self.__class__.__name__} timed out."
            raise TimeoutError(msg) from error

        except httpx.HTTPStatusError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            raise SearchApiError(msg) from e

        except httpx.HTTPError as e:
            msg = f"An error occurred while querying {self.__class__.__name__}. Error: {e}"
            raise SearchApiError(msg) from e

        documents, links = self._parse_response(response)

        logger.debug(
            "SearchApi returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    def _prepare_request(self, query: str) -> tuple[dict[str, Any], dict[str, str]]:
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""
        payload = {"q": query_prepend + " " + query, **self.search_params}
        headers = {"Authorization": f"Bearer {self.api_key.resolve_value()}", "X-SearchApi-Source": "Haystack"}
        return payload, headers

    @staticmethod
    def _parse_response(response: httpx.Response) -> tuple[list[Document], list[str]]:
        json_result = response.json()

        # AI answer engines (google_ai_mode, perplexity, chatgpt, gemini, bing_copilot) return a
        # synthesized answer instead of organic results
        if any(key in json_result for key in ("text_blocks", "markdown", "reference_links")):
            return SearchApiWebSearch._parse_ai_answer(json_result)

        # organic results are the main results from the search engine
        organic_results = []
        if "organic_results" in json_result:
            for result in json_result["organic_results"]:
                organic_results.append(
                    Document.from_dict({"title": result["title"], "content": result["snippet"], "link": result["link"]})
                )

        # answer box has a direct answer to the query
        answer_box = []
        if "answer_box" in json_result:
            answer_box = [
                Document.from_dict(
                    {
                        "title": json_result["answer_box"].get("title", ""),
                        "content": json_result["answer_box"].get("answer", ""),
                        "link": json_result["answer_box"].get("link", ""),
                    }
                )
            ]

        knowledge_graph = []
        if "knowledge_graph" in json_result:
            knowledge_graph = [
                Document.from_dict(
                    {
                        "title": json_result["knowledge_graph"].get("title", ""),
                        "content": json_result["knowledge_graph"].get("description", ""),
                    }
                )
            ]

        related_questions = []
        if "related_questions" in json_result:
            for result in json_result["related_questions"]:
                related_questions.append(
                    Document.from_dict(
                        {
                            "title": result["question"],
                            "content": result["answer"] if result.get("answer") else result.get("answer_highlight", ""),
                            "link": result.get("source", {}).get("link", ""),
                        }
                    )
                )

        documents = answer_box + knowledge_graph + organic_results + related_questions

        links = [result["link"] for result in json_result.get("organic_results", [])]
        return documents, links

    @staticmethod
    def _parse_ai_answer(json_result: dict[str, Any]) -> tuple[list[Document], list[str]]:
        """
        Parse the shared AI answer shape returned by SearchApi's answer engines.

        Engines such as `google_ai_mode`, `perplexity`, `chatgpt`, `gemini` and `bing_copilot` return the answer
        as `markdown` plus typed `text_blocks`, the sources it cites as `reference_links`, and for Google AI Mode
        the `web_results` it drew from. The answer becomes the first document, each reference link a document
        after it, followed by any web results. Links are the reference links, then the web result links.
        """
        parameters = json_result.get("search_parameters", {})
        documents: list[Document] = []

        answer = json_result.get("markdown") or _text_blocks_to_markdown(json_result.get("text_blocks", []))
        if answer:
            documents.append(
                Document.from_dict(
                    {
                        "title": parameters.get("q", ""),
                        "content": answer,
                        "engine": parameters.get("engine", ""),
                        "type": "ai_answer",
                    }
                )
            )

        links: list[str] = []
        for result in json_result.get("reference_links", []) or []:
            documents.append(
                Document.from_dict(
                    {
                        "title": result.get("title", ""),
                        "content": result.get("snippet", ""),
                        "link": result.get("link", ""),
                        "source": result.get("source", ""),
                    }
                )
            )
            if result.get("link"):
                links.append(result["link"])

        for result in json_result.get("web_results", []) or []:
            documents.append(
                Document.from_dict(
                    {
                        "title": result.get("title", ""),
                        "content": result.get("snippet", ""),
                        "link": result.get("link", ""),
                    }
                )
            )
            if result.get("link") and result["link"] not in links:
                links.append(result["link"])

        return documents, links


def _text_blocks_to_markdown(blocks: list[dict[str, Any]], depth: int = 0) -> str:
    """Render SearchApi `text_blocks` as Markdown, for responses that carry no `markdown` field."""
    lines: list[str] = []
    indent = "  " * depth
    for block in blocks or []:
        kind = block.get("type")
        answer = block.get("answer")
        if kind == "header" and answer:
            lines.append(f"## {answer}")
        elif kind in ("paragraph", None) and answer:
            lines.append(f"{indent}{answer}")
        elif kind in ("unordered_list", "ordered_list"):
            lines.extend(_list_block_lines(block, kind, indent, depth))
        elif kind == "table":
            lines.extend(_table_block_lines(block.get("table") or {}))
        elif kind == "code_blocks" and block.get("code"):
            lines.append(f"```{block.get('language') or ''}\n{block['code']}\n```")
    return "\n".join(lines)


def _list_block_lines(block: dict[str, Any], kind: str, indent: str, depth: int) -> list[str]:
    lines: list[str] = []
    if block.get("answer"):
        lines.append(f"{indent}{block['answer']}")
    for position, item in enumerate(block.get("items") or [], start=1):
        marker = f"{position}." if kind == "ordered_list" else "-"
        lines.append(f"{indent}{marker} {item.get('answer', '')}".rstrip())
        if item.get("items"):
            lines.append(_text_blocks_to_markdown(item["items"], depth + 1))
    return lines


def _table_block_lines(table: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    headers = table.get("headers") or []
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
    for row in table.get("rows") or []:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines
