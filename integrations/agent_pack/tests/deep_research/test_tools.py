from haystack import Document
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool, PipelineTool

from haystack_integrations.agent_pack.deep_research.tools import (
    ContentGate,
    TavilyWebSearchTool,
    _format_search_results,
    _read_url_result,
    make_read_url_tool,
    think_tool,
)


def test_format_search_results_without_documents():
    assert _format_search_results([]) == "No results."


def test_format_search_results_includes_title_url_and_snippet():
    docs = [
        Document(content="  first snippet  ", meta={"title": "First", "url": "https://a.example/1"}),
        Document(content="second snippet", meta={"title": "Second", "url": "https://b.example/2"}),
    ]
    out = _format_search_results(docs)
    assert "- First\n  URL: https://a.example/1\n  first snippet" in out
    assert "- Second\n  URL: https://b.example/2\n  second snippet" in out


def test_format_search_results_uses_untitled_fallback():
    out = _format_search_results([Document(content="x", meta={"url": "https://a.example"})])
    assert "- Untitled" in out


def test_content_gate_forwards_documents_with_text():
    docs = [Document(content="has text")]
    assert ContentGate().run(documents=docs) == {"documents": docs}


def test_content_gate_marks_empty_when_no_text():
    assert ContentGate().run(documents=[]) == {"empty": True}
    assert ContentGate().run(documents=[Document(content=None)]) == {"empty": True}


def test_read_url_result_returns_summary_text():
    result = {"replies": [ChatMessage.from_assistant("the summary")]}
    assert _read_url_result(result) == "the summary"


def test_read_url_result_reports_unsupported_content_type():
    result = {"replies": [], "unclassified": [Document(content="x", meta={"content_type": "image/png"})]}
    assert _read_url_result(result) == (
        "Page not read: unsupported content type image/png; only HTML and PDF are read."
    )


def test_read_url_result_reports_no_readable_text():
    assert _read_url_result({}) == "Page not read: the page returned no readable text."


def test_tavily_web_search_tool_configures_component_tool():
    tool = TavilyWebSearchTool(top_k=7)
    assert isinstance(tool, ComponentTool)
    assert tool.name == "web_search"
    assert tool._component.top_k == 7


def test_tavily_web_search_tool_serde():
    tool = TavilyWebSearchTool(top_k=7)
    data = tool.to_dict()
    assert data == {
        "type": "haystack_integrations.agent_pack.deep_research.tools.TavilyWebSearchTool",
        "data": {"top_k": 7},
    }

    restored = TavilyWebSearchTool.from_dict(data)
    assert isinstance(restored, TavilyWebSearchTool)
    assert restored.name == "web_search"
    assert restored._component.top_k == 7


def test_make_read_url_tool_builds_pipeline_tool():
    tool = make_read_url_tool(summarizer_llm=MockChatGenerator(), max_content_length=1234)
    assert isinstance(tool, PipelineTool)
    assert tool.name == "read_url"
    assert tool._input_mapping == {"urls": ["fetcher.urls"], "question": ["builder.question"]}
    components = set(tool._pipeline.graph.nodes)
    assert {"fetcher", "router", "html", "pdf", "joiner", "content_gate", "builder", "summarizer"} <= components


def test_think_tool_records_reflection():
    assert think_tool.function(reflection="still missing pricing data") == "Reflection noted"
