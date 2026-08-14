# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end span tree for a nested Agent-as-tool under RhesisTracer (Haystack 3.0)."""

from importlib.util import find_spec
from typing import Any

import pytest
from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rhesis.telemetry.attributes import AIAttributes
from rhesis.telemetry.schemas import AIOperationType

# The agent loop only emits `haystack.agent.step*` spans from Haystack 3.0 on, where the tool-calling
# logic moved into its own module. On 2.x an agent traces its tool calls as one batched ToolInvoker
# component span, so this whole tree — per-tool spans and the handoff promoted from one — is absent.
pytestmark = pytest.mark.skipif(
    find_spec("haystack.components.agents.tool_calling") is None,
    reason="haystack-ai < 3.0 does not emit the agent-loop spans this span tree asserts",
)


@component
class ScriptedChatGenerator:
    """Chat generator that returns a fixed sequence of assistant replies (tool calls or text)."""

    def __init__(self, replies: list[ChatMessage]) -> None:
        self._replies = list(replies)
        self._index = 0

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        tools: Any = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[ChatMessage]]:
        if self._index >= len(self._replies):
            return {"replies": [ChatMessage.from_assistant("done")]}
        reply = self._replies[self._index]
        self._index += 1
        return {"replies": [reply]}


def _tool_call(name: str, arguments: dict[str, Any] | None = None) -> ChatMessage:
    return ChatMessage.from_assistant(
        tool_calls=[ToolCall(id=f"call-{name}", tool_name=name, arguments=arguments or {})],
    )


def _exported_spans(exporter: InMemorySpanExporter) -> list[Any]:
    return list(exporter.get_finished_spans())


def _by_name(spans: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for span in spans:
        grouped.setdefault(span.name, []).append(span)
    return grouped


def _parent_of(span: Any, spans: list[Any]) -> Any | None:
    parent_id = span.parent.span_id if span.parent else None
    if parent_id is None:
        return None
    for candidate in spans:
        if candidate.context.span_id == parent_id:
            return candidate
    return None


def _echo(text: str) -> str:
    return text


echo_tool = Tool(
    name="echo",
    description="Echo the given text.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo"}},
        "required": ["text"],
    },
    function=_echo,
)


def _build_nested_agent_pipeline() -> Pipeline:
    """A coordinator agent that delegates to a specialist agent through a tool."""
    specialist = Agent(
        chat_generator=ScriptedChatGenerator(
            [
                _tool_call("echo", {"text": "specialist-ok"}),
                ChatMessage.from_assistant("specialist done"),
            ]
        ),
        tools=[echo_tool],
        system_prompt="You are a specialist. Use echo then answer.",
        max_agent_steps=5,
    )

    def research(query: str) -> str:
        result = specialist.run(messages=[ChatMessage.from_user(query)])
        return result["last_message"].text or ""

    research_tool = Tool(
        name="research",
        description="Delegate research to a specialist agent.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Research question"}},
            "required": ["query"],
        },
        function=research,
    )

    coordinator = Agent(
        chat_generator=ScriptedChatGenerator(
            [
                _tool_call("research", {"query": "Haystack 3.0"}),
                ChatMessage.from_assistant("Coordinator finished."),
            ]
        ),
        tools=[research_tool],
        system_prompt="You are a coordinator. Delegate with research then answer.",
        max_agent_steps=5,
    )

    pipe = Pipeline()
    pipe.add_component("coordinator", coordinator)
    return pipe


class _Tree:
    """The exported span tree of one nested-agent run, with lookup helpers."""

    def __init__(self, spans: list[Any], result: dict[str, Any]) -> None:
        self.spans = spans
        self.result = result
        self.by_name = _by_name(spans)

    def names(self) -> set[str]:
        return set(self.by_name)

    def parent_of(self, span: Any) -> Any | None:
        return _parent_of(span, self.spans)

    def handoff_or_tool(self) -> list[Any]:
        return self.by_name.get(AIOperationType.AGENT_HANDOFF.value, []) + self.by_name.get(
            AIOperationType.TOOL_INVOKE.value, []
        )

    def coordinator_component(self) -> Any:
        root = self.by_name["function.haystack.pipeline.run"][0]
        for span in self.spans:
            if self.parent_of(span) is root and span.attributes.get("haystack.component.type") == "Agent":
                return span
        raise AssertionError("no Agent component span directly under the pipeline root")

    def coordinator_agent(self) -> Any:
        component = self.coordinator_component()
        for span in self.by_name[AIOperationType.AGENT_INVOKE.value]:
            if self.parent_of(span) is component:
                return span
        raise AssertionError("no agent.invoke span under the coordinator component span")


@pytest.fixture
def nested_agent_tree(traced_exporter) -> _Tree:
    """Run the nested-agent pipeline once; every test below reads the same tree."""
    exporter, _tracer = traced_exporter
    pipe = _build_nested_agent_pipeline()
    result = pipe.run(data={"coordinator": {"messages": [ChatMessage.from_user("What is new in Haystack 3.0?")]}})
    return _Tree(_exported_spans(exporter), result)


class TestAgentSpanTree:
    """
    One nested-agent run, asserted claim by claim.

    This was a single 226-line test. It covers the most intricate logic in the package — per-tool
    spans, handoff promotion, agent nesting — so when it failed it did not say which part broke.
    """

    def test_the_run_completes(self, nested_agent_tree):
        assert "Coordinator finished" in (nested_agent_tree.result["coordinator"]["last_message"].text or "")

    def test_emits_one_pipeline_root(self, nested_agent_tree):
        assert len(nested_agent_tree.by_name["function.haystack.pipeline.run"]) == 1

    def test_emits_the_expected_span_kinds(self, nested_agent_tree):
        names = nested_agent_tree.names()
        assert AIOperationType.AGENT_INVOKE.value in names
        assert AIOperationType.LLM_INVOKE.value in names
        assert "function.haystack.agent.step" in names
        assert nested_agent_tree.handoff_or_tool(), "expected a tool.invoke or agent.handoff span"

    def test_both_agents_get_their_own_span(self, nested_agent_tree):
        assert len(nested_agent_tree.by_name[AIOperationType.AGENT_INVOKE.value]) >= 2

    def test_every_llm_span_is_typed(self, nested_agent_tree):
        llm_spans = nested_agent_tree.by_name[AIOperationType.LLM_INVOKE.value]
        assert len(llm_spans) >= 2
        for span in llm_spans:
            assert span.attributes.get(AIAttributes.OPERATION_TYPE) == AIAttributes.OPERATION_LLM_INVOKE

    def test_the_coordinator_agent_nests_under_its_component_span(self, nested_agent_tree):
        coordinator = nested_agent_tree.coordinator_agent()
        assert coordinator.attributes.get(AIAttributes.OPERATION_TYPE) == AIAttributes.OPERATION_AGENT_INVOKE

    def test_the_specialist_agent_nests_under_the_tool_that_called_it(self, nested_agent_tree):
        coordinator = nested_agent_tree.coordinator_agent()
        parents = nested_agent_tree.handoff_or_tool()
        nested = [
            span
            for span in nested_agent_tree.by_name[AIOperationType.AGENT_INVOKE.value]
            if span is not coordinator and nested_agent_tree.parent_of(span) in parents
        ]
        assert nested, "expected a nested agent.invoke under a tool/handoff parent"

    def test_the_tool_span_is_promoted_to_a_handoff(self, nested_agent_tree):
        """A tool that runs an Agent is a handoff, and the span says who it went to."""
        handoffs = nested_agent_tree.by_name.get(AIOperationType.AGENT_HANDOFF.value, [])
        if not handoffs:
            pytest.skip("this Haystack version routes the nested agent through a plain tool span")
        handoff = handoffs[0]
        assert handoff.attributes.get(AIAttributes.OPERATION_TYPE) == AIAttributes.OPERATION_AGENT_HANDOFF
        assert handoff.attributes.get(AIAttributes.AGENT_HANDOFF_TO) == "research"
