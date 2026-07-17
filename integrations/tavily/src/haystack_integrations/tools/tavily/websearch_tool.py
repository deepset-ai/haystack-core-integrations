# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.websearch.tavily import TavilyWebSearch

_DEFAULT_DESCRIPTION = (
    "Search the web. Returns the top results, each with title, exact URL and a content "
    "snippet. Cite the exact URLs verbatim."
)


def _format_search_results(documents: list[Document]) -> str:
    """
    Format the search results as the tool-result string: title + exact URL + snippet.

    :param documents: Documents returned by `TavilyWebSearch`.
    :returns: Formatted results, or a message when no results were found.
    """
    if not documents:
        return "No results."
    blocks = []
    for d in documents:
        url = d.meta.get("url", "")
        title = d.meta.get("title", "Untitled")
        snippet = (d.content or "").strip()
        blocks.append(f"- {title}\n  URL: {url}\n  {snippet}")
    return "\n".join(blocks)


class TavilyWebSearchTool(ComponentTool):
    """
    A tool that searches the web with Tavily.

    Wraps the `TavilyWebSearch` component and formats its results as a string that an LLM can cite.
    The tool parameters are derived from the component's `run` method, so the LLM can pass a `query` and,
    optionally, `search_params` overriding the ones set at initialization time.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.tools.tavily import TavilyWebSearchTool

    web_search = TavilyWebSearchTool(top_k=5, search_params={"search_depth": "advanced"})

    agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-5-mini"), tools=[web_search])

    result = agent.run(messages=[ChatMessage.from_user("What is Haystack by deepset?")])
    print(result["last_message"].text)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret | None = None,
        top_k: int | None = None,
        search_params: dict[str, Any] | None = None,
        name: str = "web_search",
        description: str = _DEFAULT_DESCRIPTION,
    ) -> None:
        """
        Initialize the TavilyWebSearchTool.

        :param api_key:
            API key for Tavily. If unset, `TavilyWebSearch` reads the `TAVILY_API_KEY` environment variable.
        :param top_k:
            Maximum number of results to return. If unset, the `TavilyWebSearch` default applies.
        :param search_params:
            Additional parameters passed to the Tavily search API.
            See the [Tavily API reference](https://docs.tavily.com/docs/tavily-api/rest_api)
            for available options. Supported keys include: `search_depth`, `include_answer`,
            `include_raw_content`, `include_domains`, `exclude_domains`.
        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params

        component_params: dict[str, Any] = {}
        if api_key is not None:
            component_params["api_key"] = api_key
        if top_k is not None:
            component_params["top_k"] = top_k
        if search_params is not None:
            component_params["search_params"] = search_params

        super().__init__(
            component=TavilyWebSearch(**component_params),
            name=name,
            description=description,
            outputs_to_string={"source": "documents", "handler": _format_search_results},
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "api_key": self.api_key.to_dict() if self.api_key else None,
                "top_k": self.top_k,
                "search_params": self.search_params,
                "name": self.name,
                "description": self.description,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TavilyWebSearchTool":
        """
        Deserialize the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        inner_data = data["data"]
        deserialize_secrets_inplace(inner_data, keys=["api_key"])
        return cls(**inner_data)
