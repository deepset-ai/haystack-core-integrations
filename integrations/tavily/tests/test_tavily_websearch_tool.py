# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.tools.tavily.websearch_tool import TavilyWebSearchTool, _format_search_results


class TestFormatSearchResults:
    def test_format_search_results(self):
        documents = [
            Document(content="Example content", meta={"title": "Example Title", "url": "https://example.com"}),
            Document(content="  Other content  ", meta={"title": "Other Title", "url": "https://other.com"}),
        ]
        formatted = _format_search_results(documents)
        assert formatted == (
            "- Example Title\n  URL: https://example.com\n  Example content\n"
            "- Other Title\n  URL: https://other.com\n  Other content"
        )

    def test_format_search_results_no_documents(self):
        assert _format_search_results([]) == "No results."

    def test_format_search_results_missing_meta(self):
        formatted = _format_search_results([Document(content=None)])
        assert formatted == "- Untitled\n  URL: \n  "


class TestTavilyWebSearchTool:
    @pytest.fixture
    def search_response(self):
        return {
            "results": [
                {"title": "Example Title", "url": "https://example.com", "content": "Example content", "score": 0.95}
            ]
        }

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        tool = TavilyWebSearchTool()

        assert tool.api_key is None
        assert tool.top_k is None
        assert tool.search_params is None
        assert tool.name == "web_search"
        assert "Search the web" in tool.description
        # the parameters exposed to the LLM are derived from the component's run method
        assert tool.parameters["required"] == ["query"]
        assert tool.parameters["properties"]["query"]["type"] == "string"
        # unset parameters fall back to the component defaults
        assert tool._component.top_k == 10
        assert tool._component.api_key.resolve_value() == "test-key"
        assert tool._component.search_params is None

    def test_init_with_params(self):
        tool = TavilyWebSearchTool(
            api_key=Secret.from_token("custom-key"),
            top_k=5,
            search_params={"search_depth": "advanced"},
            name="custom_search",
            description="Custom description.",
        )

        assert tool.api_key == Secret.from_token("custom-key")
        assert tool.top_k == 5
        assert tool.search_params == {"search_depth": "advanced"}
        assert tool.name == "custom_search"
        assert tool.description == "Custom description."
        assert tool._component.api_key == Secret.from_token("custom-key")
        assert tool._component.top_k == 5
        assert tool._component.search_params == {"search_depth": "advanced"}

    def test_to_dict(self):
        tool = TavilyWebSearchTool(
            api_key=Secret.from_env_var("TAVILY_API_KEY"),
            top_k=5,
            search_params={"search_depth": "advanced"},
        )
        data = tool.to_dict()

        assert data["type"] == "haystack_integrations.tools.tavily.websearch_tool.TavilyWebSearchTool"
        assert data["data"] == {
            "api_key": {"env_vars": ["TAVILY_API_KEY"], "strict": True, "type": "env_var"},
            "top_k": 5,
            "search_params": {"search_depth": "advanced"},
            "name": "web_search",
            "description": tool.description,
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.tools.tavily.websearch_tool.TavilyWebSearchTool",
            "data": {
                "api_key": {"env_vars": ["TAVILY_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 3,
                "search_params": {"include_answer": True},
                "name": "custom_search",
                "description": "Custom description.",
            },
        }
        tool = TavilyWebSearchTool.from_dict(data)

        assert tool.api_key == Secret.from_env_var("TAVILY_API_KEY")
        assert tool.top_k == 3
        assert tool.search_params == {"include_answer": True}
        assert tool.name == "custom_search"
        assert tool.description == "Custom description."
        assert tool._component.top_k == 3

    def test_invoke(self, search_response):
        tool = TavilyWebSearchTool(api_key=Secret.from_token("test-key"), top_k=5)
        mock_client = MagicMock()
        mock_client.search.return_value = search_response
        tool._component._tavily_client = mock_client

        result = tool.invoke(query="test query")

        assert result["documents"] == [
            Document(
                content="Example content",
                meta={"title": "Example Title", "url": "https://example.com"},
                score=0.95,
            )
        ]
        mock_client.search.assert_called_once_with(query="test query", max_results=5)

    def test_invoke_outputs_to_string(self, search_response):
        tool = TavilyWebSearchTool(api_key=Secret.from_token("test-key"))
        mock_client = MagicMock()
        mock_client.search.return_value = search_response
        tool._component._tavily_client = mock_client

        result = tool.invoke(query="test query")
        message = tool.outputs_to_string["handler"](result[tool.outputs_to_string["source"]])

        assert message == "- Example Title\n  URL: https://example.com\n  Example content"
