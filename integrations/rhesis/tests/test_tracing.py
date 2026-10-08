# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests against a live Rhesis backend.

The exporter logs and swallows every failure (bad key, wrong URL, rejected payload), so a run that
completes proves nothing. These tests read each trace back from the Rhesis API and assert on what
the backend actually stored.
"""

import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from rhesis.telemetry.attributes import AIAttributes
from rhesis.telemetry.constants import ConversationContext

from haystack_integrations.components.connectors.rhesis import RhesisConnector
from haystack_integrations.tracing.rhesis import DEFAULT_TURN_SPAN_NAME, RhesisTracing, rhesis_invocation_context

RHESIS_CLOUD_URL = "https://api.rhesis.ai"

# Ingestion is acknowledged before post-processing, so a trace can take a moment to become readable.
TRACE_POLL_TIMEOUT_S = 60
TRACE_POLL_INTERVAL_S = 2

requires_api_key = pytest.mark.skipif(
    not os.environ.get("RHESIS_API_KEY"),
    reason="Missing required environment variable: RHESIS_API_KEY",
)


@component
class StubChatGenerator:
    """Minimal ChatGenerator-shaped component, so the tests need no LLM API key."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        reply = ChatMessage.from_assistant(
            "Berlin is the capital of Germany.",
            meta={
                "model": "stub-model",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        )
        return {"replies": [reply]}


@component
class ScriptedChatGenerator:
    """Replays a fixed list of replies, so an Agent takes a deterministic tool-calling path."""

    def __init__(self, replies: list[ChatMessage]) -> None:
        self._replies = list(replies)

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, list[ChatMessage]]:
        reply = self._replies.pop(0) if self._replies else ChatMessage.from_assistant("done")
        return {"replies": [reply]}


class RhesisAPI:
    """Reads traces back through the same API the Rhesis UI uses."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def list_traces(self, **params: Any) -> list[dict[str, Any]]:
        response = self._session.get(f"{self.base_url}/telemetry/traces", params=params, timeout=30)
        response.raise_for_status()
        return response.json()["traces"]

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        """
        Poll until the trace is stored, then return its full span tree.

        The detail endpoint requires a `project_id`. A project-scoped key (the only kind that can
        ingest without one) scopes the list endpoint to its project, so the summary supplies it.
        """
        deadline = time.monotonic() + TRACE_POLL_TIMEOUT_S
        while True:
            traces = self.list_traces(search=trace_id, limit=10)
            summary = next((t for t in traces if t["trace_id"] == trace_id), None)
            if summary is not None:
                break
            if time.monotonic() > deadline:
                msg = f"Trace {trace_id} did not appear in Rhesis within {TRACE_POLL_TIMEOUT_S}s"
                raise AssertionError(msg)
            time.sleep(TRACE_POLL_INTERVAL_S)

        response = self._session.get(
            f"{self.base_url}/telemetry/traces/{trace_id}", params={"project_id": summary["project_id"]}, timeout=30
        )
        response.raise_for_status()
        return response.json()


def _walk(spans: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for span in spans:
        yield span
        yield from _walk(span.get("children", []))


def _spans_named(trace: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [span for span in _walk(trace["root_spans"]) if span["span_name"] == name]


@pytest.fixture
def base_url() -> str:
    return os.environ.get("RHESIS_BASE_URL") or RHESIS_CLOUD_URL


@pytest.fixture
def rhesis_api(base_url) -> RhesisAPI:
    return RhesisAPI(base_url, os.environ["RHESIS_API_KEY"])


@pytest.fixture
def environment() -> str:
    # Unique per test, so traces from CI are easy to tell apart from anything else in the project.
    return f"haystack-it-{uuid.uuid4().hex[:8]}"


@requires_api_key
@pytest.mark.integration
class TestLiveBackend:
    def test_pipeline_trace_is_stored(self, rhesis_api, base_url, environment):
        session_id = f"sess-{uuid.uuid4().hex}"

        pipe = Pipeline()
        pipe.add_component("tracer", RhesisConnector("Chat example", base_url=base_url, environment=environment))
        pipe.add_component("prompt_builder", ChatPromptBuilder())
        pipe.add_component("llm", StubChatGenerator())
        pipe.connect("prompt_builder.prompt", "llm.messages")

        response = pipe.run(
            data={
                "prompt_builder": {
                    "template_variables": {"location": "Berlin"},
                    "template": [ChatMessage.from_user("Tell me about {{location}}")],
                },
                "tracer": {"invocation_context": {"session_id": session_id}},
            }
        )
        trace_id = response["tracer"]["trace_id"]
        assert len(trace_id) == 32

        trace = rhesis_api.get_trace(trace_id)

        assert trace["environment"] == environment
        assert trace["error_count"] == 0
        # Pipeline root plus one span per component: tracer, prompt_builder, llm.
        assert trace["span_count"] >= 4
        assert trace["total_input_tokens"] == 12
        assert trace["total_output_tokens"] == 8

        [root] = trace["root_spans"]
        assert root["span_name"] == "function.haystack.pipeline.run"
        assert root["attributes"][AIAttributes.SESSION_ID] == session_id
        attrs = ConversationContext.SpanAttributes
        # No conversation input: the question enters as a prompt template, and the rendered prompt
        # never appears in the pipeline's own input, which is all the tracer reads it from.
        assert root["attributes"][attrs.CONVERSATION_OUTPUT] == "Berlin is the capital of Germany."

        [llm] = _spans_named(trace, "ai.llm.invoke")
        assert llm["attributes"][AIAttributes.MODEL_NAME] == "stub-model"
        assert llm["attributes"][AIAttributes.LLM_TOKENS_TOTAL] == 20

    def test_standalone_agent_trace_is_stored(self, rhesis_api, base_url, environment):
        pytest.importorskip(
            "haystack.components.agents.tool_calling",
            reason="haystack-ai < 3.0 does not emit the agent-loop spans this asserts",
        )
        from haystack.components.agents import Agent  # noqa: PLC0415

        def echo(text: str) -> str:
            return text

        echo_tool = Tool(
            name="echo",
            description="Echo the given text.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=echo,
        )
        agent = Agent(
            chat_generator=ScriptedChatGenerator(
                [
                    ChatMessage.from_assistant(
                        tool_calls=[ToolCall(id="c1", tool_name="echo", arguments={"text": "hello"})]
                    ),
                    ChatMessage.from_assistant("All done."),
                ]
            ),
            tools=[echo_tool],
            system_prompt="Use echo, then answer.",
            max_agent_steps=5,
        )
        # Constructing the connector installs the tracer; the agent runs without a pipeline.
        connector = RhesisConnector("Agent example", base_url=base_url, environment=environment)
        session_id = f"sess-{uuid.uuid4().hex}"

        with rhesis_invocation_context({"session_id": session_id}):
            agent.run(messages=[ChatMessage.from_user("Say hello")])
        connector.tracer.flush()

        # Without a pipeline there is no connector output to read the trace ID from, and the root
        # span is already closed; the per-test environment identifies the trace instead.
        [summary] = rhesis_api.list_traces(environment=environment, limit=10)
        trace = rhesis_api.get_trace(summary["trace_id"])

        [root] = trace["root_spans"]
        assert root["span_name"] == "ai.agent.invoke"
        assert root["attributes"][AIAttributes.SESSION_ID] == session_id
        assert len(_spans_named(trace, "ai.llm.invoke")) == 2
        assert len(_spans_named(trace, "ai.tool.invoke")) == 1

    def test_conversation_turns_share_one_trace(self, rhesis_api, base_url, environment):
        pipe = Pipeline()
        pipe.add_component("llm", StubChatGenerator())

        tracing = RhesisTracing("Conversation example", base_url=base_url, environment=environment)
        assert tracing.enabled
        conversation_id = f"conv-{uuid.uuid4().hex}"
        tracing.start_conversation(conversation_id)

        trace_ids = set()
        for message in ["Hello", "Tell me more"]:
            with tracing.turn(message) as turn:
                result = pipe.run({"llm": {"messages": [ChatMessage.from_user(message)]}})
                turn.output = result["llm"]["replies"][0].text
                trace_ids.add(format(turn.span.get_span_context().trace_id, "032x"))
        tracing.flush()

        [trace_id] = trace_ids
        trace = rhesis_api.get_trace(trace_id)

        assert trace["conversation_id"] == conversation_id
        turns = _spans_named(trace, DEFAULT_TURN_SPAN_NAME)
        attrs = ConversationContext.SpanAttributes
        assert sorted(t["attributes"][attrs.CONVERSATION_INPUT] for t in turns) == ["Hello", "Tell me more"]
        assert len(_spans_named(trace, "function.haystack.pipeline.run")) == 2
