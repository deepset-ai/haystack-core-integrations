import os

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.core.serialization import allow_deserialization_module
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool

from haystack_integrations.agent_pack.deep_research.agent import (
    _collect_note,
    _default_llm,
    _make_researcher_agent,
    create_deep_research_agent,
)
from haystack_integrations.agent_pack.deep_research.hooks import ScopeHook, WriteHook


def _delegate_then_stop(subtopic):
    return [
        ChatMessage.from_assistant(
            tool_calls=[
                ToolCall(tool_name="research_subtopic", arguments={"messages": [{"role": "user", "content": subtopic}]})
            ]
        ),
        ChatMessage.from_assistant("research complete"),
    ]


def test_default_llm_applies_pack_settings():
    llm = _default_llm("gpt-5.4-mini")
    assert isinstance(llm, OpenAIResponsesChatGenerator)
    assert llm.model == "gpt-5.4-mini"
    assert llm.timeout == 180.0
    assert llm.max_retries == 5


def test_collect_note_appends_message_text():
    assert _collect_note(None, ChatMessage.from_assistant("first")) == ["first"]
    assert _collect_note(["first"], ChatMessage.from_assistant("second")) == ["first", "second"]


def test_collect_note_skips_empty_message():
    assert _collect_note(["kept"], ChatMessage.from_assistant("")) == ["kept"]


def test_make_researcher_agent_wires_the_three_tools():
    researcher = _make_researcher_agent(
        researcher_llm=MockChatGenerator(),
        page_summary_llm=MockChatGenerator(),
        search_tools=None,
        max_researcher_steps=12,
        max_page_chars=1000,
    )
    assert isinstance(researcher, Agent)
    assert [t.name for t in researcher.tools] == ["web_search", "read_url", "think_tool"]
    assert researcher.exit_conditions == ["text"]
    assert researcher.max_agent_steps == 12


def test_create_deep_research_agent():
    llm = MockChatGenerator()
    brief_llm = MockChatGenerator()
    researcher_llm = MockChatGenerator()
    page_summary_llm = MockChatGenerator()
    report_llm = MockChatGenerator()
    agent = create_deep_research_agent(
        llm=llm,
        brief_llm=brief_llm,
        researcher_llm=researcher_llm,
        page_summary_llm=page_summary_llm,
        report_llm=report_llm,
        max_subtopics=3,
        max_concurrent_researchers=2,
        max_agent_steps=6,
    )
    assert [t.name for t in agent.tools] == ["research_subtopic", "think_tool"]
    assert agent.exit_conditions == ["text"]
    assert agent.max_agent_steps == 6
    assert agent.tool_concurrency_limit == 2
    assert set(agent.state_schema) >= {"notes", "brief", "report"}
    assert "at most 3 focused sub-questions" in agent.system_prompt
    assert [type(h) for h in agent.hooks["before_run"]] == [ScopeHook]
    assert [type(h) for h in agent.hooks["after_run"]] == [WriteHook]
    assert agent.chat_generator is llm
    assert agent.hooks["before_run"][0].generator is brief_llm
    assert agent.hooks["after_run"][0].generator is report_llm
    assert agent.tools[0]._component.chat_generator is researcher_llm

    default_agent = create_deep_research_agent()
    assert isinstance(default_agent.chat_generator, OpenAIResponsesChatGenerator)
    assert default_agent.chat_generator.model == "gpt-5.4"


def test_custom_search_tools_and_system_prompt_bypass_tavily(monkeypatch):
    class _FailedTavilyImport:
        def check(self):
            msg = "Tavily should not be checked for custom search tools"
            raise AssertionError(msg)

    @tool
    def custom_search(query: str) -> str:
        """Search a custom source."""
        return query

    monkeypatch.setattr("haystack_integrations.agent_pack.deep_research.agent.tavily_import", _FailedTavilyImport())
    agent = create_deep_research_agent(
        llm=MockChatGenerator(),
        brief_llm=MockChatGenerator(),
        researcher_llm=MockChatGenerator(),
        page_summary_llm=MockChatGenerator(),
        report_llm=MockChatGenerator(),
        search_tools=[custom_search],
        system_prompt="Research at most {{ max_subtopics }} topics.",
        max_subtopics=2,
    )

    researcher = agent.tools[0]._component
    assert [tool.name for tool in researcher.tools] == ["custom_search", "read_url", "think_tool"]
    assert agent.system_prompt == "Research at most 2 topics."


def test_deep_research_run(monkeypatch):
    # agent.run() warms up the TavilyWebSearch tool component, which resolves TAVILY_API_KEY
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    agent = create_deep_research_agent(
        brief_llm=MockChatGenerator("Brief: investigate the causes of X."),
        llm=MockChatGenerator(responses=_delegate_then_stop("What causes X?")),
        researcher_llm=MockChatGenerator("X is caused by Y (https://example.com/x).\nSources:\nhttps://example.com/x"),
        page_summary_llm=MockChatGenerator(),
        report_llm=MockChatGenerator("# Report\nX is caused by Y (https://example.com/x)."),
        max_subtopics=1,
    )
    result = agent.run(messages=[ChatMessage.from_user("Tell me about X")])

    assert result["brief"] == "Brief: investigate the causes of X."
    assert result["notes"] == ["X is caused by Y (https://example.com/x).\nSources:\nhttps://example.com/x"]
    assert result["report"] == "# Report\nX is caused by Y (https://example.com/x)."
    assert result["last_message"].text == result["report"]


def test_create_deep_research_agent_serialization_roundtrip():
    allow_deserialization_module("haystack_integrations.agent_pack.*")
    agent = create_deep_research_agent(
        brief_llm=MockChatGenerator(),
        llm=MockChatGenerator(),
        researcher_llm=MockChatGenerator(),
        page_summary_llm=MockChatGenerator(),
        report_llm=MockChatGenerator(),
    )

    restored = Agent.from_dict(agent.to_dict())

    assert isinstance(restored, Agent)
    assert [t.name for t in restored.tools] == ["research_subtopic", "think_tool"]
    assert [type(h) for h in restored.hooks["before_run"]] == [ScopeHook]
    assert [type(h) for h in restored.hooks["after_run"]] == [WriteHook]
    assert set(restored.state_schema) >= {"notes", "brief", "report"}


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("TAVILY_API_KEY")),
    reason="OPENAI_API_KEY and TAVILY_API_KEY required",
)
def test_deep_research_live_run():
    agent = create_deep_research_agent(max_subtopics=2, max_agent_steps=4, max_researcher_steps=6)
    result = agent.run(messages=[ChatMessage.from_user("What are the main causes and effects of ocean acidification?")])

    assert result["brief"].strip()
    assert "acidification" in result["brief"].lower()
    assert result["notes"] and all(note.strip() for note in result["notes"])
    report = result["report"]
    assert report.strip()
    assert "http" in report
    assert result["last_message"].text == report
    assert result["tool_call_counts"]["research_subtopic"] >= 1
    assert result["tool_call_counts"]["think_tool"] >= 1
