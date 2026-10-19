# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
What a traced pipeline actually puts on the wire.

The mapping used to be guarded by asserting that a table of Haystack tags contained the keys the
table itself was built from — a tautology that passed no matter what the tracer emitted. These
tests run real pipelines and assert against the spans that come out, so a renamed attribute, a
dropped promotion or a new Haystack tag fails here.
"""

# Imports inside test bodies are deliberate here: the agent-loop modules only exist on
# haystack 3.0 and sit behind an importorskip, and the async entry point differs by version.
# ruff: noqa: PLC0415

from typing import Any

import pytest
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from rhesis.telemetry.attributes import AIAttributes
from rhesis.telemetry.constants import ConversationContext

from haystack_integrations.tracing.rhesis import rhesis_invocation_context

# Every attribute key the integration promises. A silent rename breaks this list, which is the
# point: these are the names the Rhesis backend indexes and the integration docs document.
PROMOTED_ATTRIBUTE_KEYS = frozenset(
    {
        AIAttributes.OPERATION_TYPE,
        AIAttributes.MODEL_NAME,
        AIAttributes.LLM_TOKENS_INPUT,
        AIAttributes.LLM_TOKENS_OUTPUT,
        AIAttributes.LLM_TOKENS_TOTAL,
        ConversationContext.SpanAttributes.CONVERSATION_INPUT,
        ConversationContext.SpanAttributes.CONVERSATION_OUTPUT,
        ConversationContext.SpanAttributes.CONVERSATION_ID,
        ConversationContext.SpanAttributes.IS_TURN_ROOT,
        AIAttributes.SESSION_ID,
    }
)


@component
class StubChatGenerator:
    """ChatGenerator-shaped, so the tracer takes the model/token enrichment path."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        reply = ChatMessage.from_assistant(
            "Berlin is the capital of Germany.",
            meta={
                "model": "stub-model",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        )
        return {"replies": [reply]}


@component
class StubRetriever:
    @component.output_types(documents=list)
    def run(self, query: str) -> dict[str, Any]:
        return {"documents": []}


@component
class StubTextEmbedder:
    @component.output_types(embedding=list, meta=dict)
    def run(self, text: str) -> dict[str, Any]:
        return {"embedding": [0.1, 0.2], "meta": {"model": "stub-embedder", "usage": {"prompt_tokens": 3}}}


def _spans_by_name(exporter) -> dict[str, Any]:
    return {span.name: span for span in exporter.get_finished_spans()}


class TestEmittedSpanNames:
    def test_rag_pipeline_span_names(self, traced_exporter):
        """The waterfall a reviewer sees for the `example/basic_rag.py` pipeline."""
        exporter, _ = traced_exporter

        pipe = Pipeline()
        pipe.add_component("embedder", StubTextEmbedder())
        pipe.add_component("retriever", StubRetriever())
        pipe.add_component("llm", StubChatGenerator())
        pipe.run(
            {
                "embedder": {"text": "What is the capital of Germany?"},
                "retriever": {"query": "What is the capital of Germany?"},
                "llm": {"messages": [ChatMessage.from_user("What is the capital of Germany?")]},
            }
        )

        names = set(_spans_by_name(exporter))
        assert "function.haystack.pipeline.run" in names
        assert "ai.embedding.generate" in names
        assert "ai.retrieval" in names
        assert "ai.llm.invoke" in names

    def test_span_names_are_strings_not_enum_reprs(self, traced_exporter):
        """
        `AIOperationType` is a `(str, Enum)`, so `str(member)` is `AIOperationType.LLM_INVOKE`.

        Equality and the OTLP encoder resolve to the value, but anything that formats `span.name`
        into text gets the wrong string, so the boundary has to hand over `.value`.
        """
        exporter, _ = traced_exporter

        pipe = Pipeline()
        pipe.add_component("llm", StubChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("hi")]}})

        for span in exporter.get_finished_spans():
            assert type(span.name) is str, f"{span.name!r} is {type(span.name).__name__}, not str"
            assert not f"{span.name}".startswith("AIOperationType.")


class TestEmittedAttributes:
    def test_llm_span_carries_model_and_tokens(self, traced_exporter):
        exporter, _ = traced_exporter

        pipe = Pipeline()
        pipe.add_component("llm", StubChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("What is the capital of Germany?")]}})

        llm = _spans_by_name(exporter)["ai.llm.invoke"]
        assert llm.attributes[AIAttributes.MODEL_NAME] == "stub-model"
        assert llm.attributes[AIAttributes.LLM_TOKENS_INPUT] == 12
        assert llm.attributes[AIAttributes.LLM_TOKENS_OUTPUT] == 8
        assert llm.attributes[AIAttributes.LLM_TOKENS_TOTAL] == 20
        assert llm.attributes[AIAttributes.OPERATION_TYPE] == AIAttributes.OPERATION_LLM_INVOKE

    def test_llm_span_records_prompt_and_completion_events(self, traced_exporter):
        exporter, _ = traced_exporter

        pipe = Pipeline()
        pipe.add_component("llm", StubChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("What is the capital of Germany?")]}})

        llm = _spans_by_name(exporter)["ai.llm.invoke"]
        event_names = [event.name for event in llm.events]
        assert "ai.prompt" in event_names
        assert "ai.completion" in event_names

    def test_root_span_promotes_the_turn(self, traced_exporter):
        exporter, _ = traced_exporter

        pipe = Pipeline()
        pipe.add_component("llm", StubChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("What is the capital of Germany?")]}})

        attrs = ConversationContext.SpanAttributes
        root = _spans_by_name(exporter)["function.haystack.pipeline.run"]
        assert root.attributes[attrs.CONVERSATION_INPUT] == "What is the capital of Germany?"
        # A ChatGenerator answers on `replies`, not `messages`. Only `messages` used to be read, so
        # the quickstart's own pipeline shape recorded the question and left the answer blank.
        assert root.attributes[attrs.CONVERSATION_OUTPUT] == "Berlin is the capital of Germany."

    @pytest.mark.parametrize("key", sorted(PROMOTED_ATTRIBUTE_KEYS))
    def test_promoted_attribute_keys_keep_their_names(self, key):
        """
        Pins the wire names themselves.

        These are what the Rhesis backend indexes on and what the mapping table in the docs advertises,
        so renaming one in the SDK has to be a deliberate, visible change here too.
        """
        assert key.startswith(("ai.", "rhesis.")), f"{key} is outside the namespaces Rhesis accepts"


class TestStandaloneAgent:
    """
    An `Agent` traced with no pipeline around it.

    This is the case the integration handles best and had no coverage at all: the connector is
    constructed and never used again, so `ai.agent.invoke` is the trace root rather than a child of
    `function.haystack.pipeline.run`.
    """

    @pytest.fixture
    def agent(self):
        pytest.importorskip(
            "haystack.components.agents.tool_calling",
            reason="haystack-ai < 3.0 does not emit the agent-loop spans this asserts",
        )
        from haystack.components.agents import Agent
        from haystack.dataclasses import ToolCall
        from haystack.tools import Tool

        def echo(text: str) -> str:
            return text

        echo_tool = Tool(
            name="echo",
            description="Echo the given text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            function=echo,
        )

        @component
        class ScriptedChatGenerator:
            def __init__(self, replies):
                self._replies = list(replies)
                self._index = 0

            @component.output_types(replies=list[ChatMessage])
            def run(self, messages, tools=None, generation_kwargs=None, **kwargs):
                if self._index >= len(self._replies):
                    return {"replies": [ChatMessage.from_assistant("done")]}
                reply = self._replies[self._index]
                self._index += 1
                return {"replies": [reply]}

        return Agent(
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

    def test_agent_invoke_is_the_trace_root(self, traced_exporter, agent):
        exporter, _ = traced_exporter
        agent.run(messages=[ChatMessage.from_user("say hello")])

        spans = exporter.get_finished_spans()
        roots = [span for span in spans if span.parent is None]
        assert len(roots) == 1
        assert roots[0].name == "ai.agent.invoke"
        assert roots[0].attributes[AIAttributes.OPERATION_TYPE] == AIAttributes.OPERATION_AGENT_INVOKE

    def test_the_loop_emits_a_span_per_step_and_per_tool_call(self, traced_exporter, agent):
        exporter, _ = traced_exporter
        agent.run(messages=[ChatMessage.from_user("say hello")])

        names = [span.name for span in exporter.get_finished_spans()]
        assert names.count("function.haystack.agent.step") >= 2
        assert "ai.llm.invoke" in names
        assert "ai.tool.invoke" in names

    def test_the_agent_root_carries_the_turn(self, traced_exporter, agent):
        exporter, _ = traced_exporter
        with rhesis_invocation_context({"session_id": "standalone-1"}):
            agent.run(messages=[ChatMessage.from_user("say hello")])

        attrs = ConversationContext.SpanAttributes
        root = next(span for span in exporter.get_finished_spans() if span.parent is None)
        assert root.attributes[attrs.CONVERSATION_ID] == "standalone-1"
        assert root.attributes[attrs.IS_TURN_ROOT] is True
        assert root.attributes[attrs.CONVERSATION_INPUT] == "say hello"
        assert root.attributes[attrs.CONVERSATION_OUTPUT] == "All done."


class TestAsyncPipeline:
    """`haystack.async_pipeline.run` was only ever exercised as a string in a mapping table."""

    @staticmethod
    def _async_pipeline() -> Any:
        """
        Build a pipeline with whichever async entry point this Haystack offers.

        3.0 folded `AsyncPipeline` into `Pipeline.run_async`; 2.x has the separate class. Chosen up
        front rather than by moving components afterwards — Haystack refuses to share a component
        instance between two pipelines.
        """
        if hasattr(Pipeline, "run_async"):
            return Pipeline()
        from haystack import AsyncPipeline

        return AsyncPipeline()

    @staticmethod
    def _run_async(pipe: Any, data: dict[str, Any]) -> None:
        import asyncio

        asyncio.run(pipe.run_async(data))

    def test_async_run_emits_exactly_one_root(self, traced_exporter):
        exporter, _ = traced_exporter

        pipe = self._async_pipeline()
        pipe.add_component("llm", StubChatGenerator())
        self._run_async(pipe, {"llm": {"messages": [ChatMessage.from_user("hi")]}})

        roots = [span for span in exporter.get_finished_spans() if span.parent is None]
        assert len(roots) == 1
        # 3.0 reports async as `haystack.pipeline.run` with an execution-mode tag; 2.x has its own
        # operation. Either is a valid root name, but it has to be one of them.
        assert roots[0].name in ("function.haystack.pipeline.run", "function.haystack.async_pipeline.run")

    def test_async_children_parent_to_the_root(self, traced_exporter):
        exporter, _ = traced_exporter

        pipe = self._async_pipeline()
        pipe.add_component("llm", StubChatGenerator())
        pipe.add_component("embedder", StubTextEmbedder())
        self._run_async(
            pipe,
            {
                "llm": {"messages": [ChatMessage.from_user("hi")]},
                "embedder": {"text": "hi"},
            },
        )

        spans = exporter.get_finished_spans()
        root = next(span for span in spans if span.parent is None)
        children = [span for span in spans if span.parent is not None]
        assert len(children) >= 2
        for child in children:
            assert child.parent.span_id == root.context.span_id

    def test_async_run_promotes_the_turn(self, traced_exporter):
        """The root-span enrichment has to work on the async path too, not only the sync one."""
        exporter, _ = traced_exporter

        pipe = self._async_pipeline()
        pipe.add_component("llm", StubChatGenerator())
        with rhesis_invocation_context({"session_id": "async-1"}):
            self._run_async(pipe, {"llm": {"messages": [ChatMessage.from_user("hi")]}})

        attrs = ConversationContext.SpanAttributes
        root = next(span for span in exporter.get_finished_spans() if span.parent is None)
        assert root.attributes[attrs.CONVERSATION_ID] == "async-1"
        assert root.attributes[attrs.CONVERSATION_INPUT] == "hi"
        assert root.attributes[attrs.CONVERSATION_OUTPUT] == "Berlin is the capital of Germany."
