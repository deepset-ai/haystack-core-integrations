# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end span tree for a nested Agent-as-tool under RhesisTracer (Haystack 3.0)."""

from importlib.util import find_spec
from typing import Any

import pytest
from haystack import Pipeline, component, tracing
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.tracing import disable_tracing
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rhesis.sdk.telemetry.attributes import AIAttributes
from rhesis.telemetry.schemas import AIOperationType

from haystack_integrations.tracing.rhesis.tracer import RhesisTelemetry, RhesisTracer

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


@pytest.fixture
def traced_exporter():
    """Install RhesisTracer with an in-memory OTel exporter for one test."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = RhesisTelemetry(
        provider=provider,
        otel_tracer=provider.get_tracer("test-agent-tree"),
        project_id="proj-test",
        environment="test",
        base_url="http://localhost:8080",
    )
    rhesis_tracer = RhesisTracer(telemetry=telemetry, name="agent-tree-test")
    rhesis_tracer.enforce_flush = False

    previous = tracing.tracer.is_content_tracing_enabled
    tracing.tracer.is_content_tracing_enabled = True
    tracing.enable_tracing(rhesis_tracer)
    try:
        yield exporter, rhesis_tracer
    finally:
        disable_tracing()
        tracing.tracer.is_content_tracing_enabled = previous
        exporter.clear()


class TestAgentSpanTree:
    def test_nested_agent_as_tool_emits_typed_span_tree(self, traced_exporter):
        exporter, _tracer = traced_exporter

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
        result = pipe.run(
            data={"coordinator": {"messages": [ChatMessage.from_user("What is new in Haystack 3.0?")]}},
        )

        assert "Coordinator finished" in (result["coordinator"]["last_message"].text or "")

        spans = _exported_spans(exporter)
        by_name = _by_name(spans)
        names = {span.name for span in spans}

        assert "function.haystack.pipeline.run" in names
        assert AIOperationType.AGENT_INVOKE in names
        assert AIOperationType.LLM_INVOKE in names
        assert AIOperationType.AGENT_HANDOFF in names or AIOperationType.TOOL_INVOKE in names
        assert "function.haystack.agent.step" in names

        agent_spans = by_name[AIOperationType.AGENT_INVOKE]
        assert len(agent_spans) >= 2  # coordinator + specialist

        llm_spans = by_name[AIOperationType.LLM_INVOKE]
        assert len(llm_spans) >= 2
        for llm_span in llm_spans:
            assert llm_span.attributes.get(AIAttributes.OPERATION_TYPE) == AIAttributes.OPERATION_LLM_INVOKE

        pipeline_roots = by_name["function.haystack.pipeline.run"]
        assert len(pipeline_roots) == 1
        pipeline_root = pipeline_roots[0]

        coordinator_component = None
        for span in spans:
            parent = _parent_of(span, spans)
            if parent is pipeline_root and span.attributes.get("haystack.component.type") == "Agent":
                coordinator_component = span
                break
        assert coordinator_component is not None

        coordinator_agent = None
        for span in agent_spans:
            if _parent_of(span, spans) is coordinator_component:
                coordinator_agent = span
                break
        assert coordinator_agent is not None
        assert coordinator_agent.attributes.get(AIAttributes.OPERATION_TYPE) == AIAttributes.OPERATION_AGENT_INVOKE

        handoff_or_tool = by_name.get(AIOperationType.AGENT_HANDOFF, []) + by_name.get(AIOperationType.TOOL_INVOKE, [])
        assert handoff_or_tool
        specialist_agents = [
            span for span in agent_spans if span is not coordinator_agent and _parent_of(span, spans) in handoff_or_tool
        ]
        assert specialist_agents, "expected nested agent.invoke under a tool/handoff parent"

        handoffs = by_name.get(AIOperationType.AGENT_HANDOFF, [])
        if handoffs:
            handoff = handoffs[0]
            assert handoff.attributes.get(AIAttributes.OPERATION_TYPE) == AIAttributes.OPERATION_AGENT_HANDOFF
            assert handoff.attributes.get(AIAttributes.AGENT_HANDOFF_TO) == "research"
