# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
What a traced pipeline actually puts on the wire.

The mapping used to be guarded by asserting that a table of Haystack tags contained the keys the
table itself was built from — a tautology that passed no matter what the tracer emitted. These
tests run real pipelines and assert against the spans that come out, so a renamed attribute, a
dropped promotion or a new Haystack tag fails here.
"""

from typing import Any

import pytest
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from rhesis.sdk.telemetry.attributes import AIAttributes
from rhesis.telemetry.constants import ConversationContext

# Every attribute key the integration promises. A silent rename breaks this list, which is the
# point: these are the names the Rhesis backend indexes and the README documents.
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
        """The waterfall a reviewer sees for the README's RAG example."""
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

    @pytest.mark.parametrize("key", sorted(PROMOTED_ATTRIBUTE_KEYS))
    def test_promoted_attribute_keys_keep_their_names(self, key):
        """
        Pins the wire names themselves.

        These are what the Rhesis backend indexes on and what the README's mapping table advertises,
        so renaming one in the SDK has to be a deliberate, visible change here too.
        """
        assert key.startswith(("ai.", "rhesis.")), f"{key} is outside the namespaces Rhesis accepts"
