import asyncio

import pytest
from haystack import Document, Pipeline, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.dataclasses import ChatMessage

from haystack_integrations.evaluation.dataclasses import ModelTokenUsage
from haystack_integrations.tracing.agent_pack import EVAL_CASE_SPAN, HarnessSpan, HarnessTracer
from haystack_integrations.tracing.agent_pack.tracer import (
    MAX_RECORDED_TEXT_CHARS,
    MAX_RECORDED_TEXTS,
    CollectedSpans,
    ReportedUsage,
    _measure_output,
    _reported_tokens,
)

TOKENS = ModelTokenUsage(input_tokens=3, output_tokens=1)


def generator_span(model="m", tokens=TOKENS, **overrides):
    """One span shaped like an LLM call that reported a single reply."""
    span = HarnessSpan(is_generator_span=True, **overrides)
    span.reported_output = True
    span.reported_usage = [ReportedUsage(model=model, tokens=tokens)]
    return span


def emit(tracer, component, output):
    """Emit one component's output the way Haystack's component tracing does."""
    with tracer.trace("haystack.component.run", tags={"haystack.component.name": component}) as span:
        span.set_content_tag("haystack.component.output", output)


class TestHarnessTracer:
    def test_usage_without_content_tracing(self):
        tracer = HarnessTracer()
        old_content = tracing.tracer.is_content_tracing_enabled
        tracing.tracer.is_content_tracing_enabled = False
        ranker = LLMRanker(
            chat_generator=MockChatGenerator(
                '{"documents": [{"index": 1}]}',
                model="ranker",
                meta={"usage": {"input_tokens": 7, "output_tokens": 2}},
            )
        )
        pipeline = Pipeline()
        pipeline.add_component("ranker", ranker)
        agent = Agent(
            chat_generator=MockChatGenerator(
                "answer",
                model="coordinator",
                meta={"usage": {"input_tokens": 11, "output_tokens": 3}},
            )
        )
        try:
            with tracer.activate(), tracing.tracer.trace(EVAL_CASE_SPAN) as span:
                pipeline.run({"ranker": {"query": "Berlin", "documents": [Document(content="Berlin")]}})
                agent.run(messages=[ChatMessage.from_user("q")])
            summary = span.collected.summarize()
            assert summary.all_tokens_reported
            assert summary.llm_calls == 2
            assert summary.models["ranker"].input_tokens == 7
            assert summary.models["coordinator"].input_tokens == 11
        finally:
            tracing.tracer.is_content_tracing_enabled = old_content

    def test_records_component_output(self):
        """The sizes and samples a summary reports come from the content tag a component emits."""
        tracer = HarnessTracer()
        with tracer.trace(EVAL_CASE_SPAN) as span:
            emit(tracer, "expander", {"queries": ["who owns it", "when was it sold"]})
        summary = span.collected.summarize()
        assert summary.outputs == {"expander": {"queries": 2}}
        assert summary.texts == {"expander": {"queries": ["who owns it", "when was it sold"]}}

    def test_concurrent_eval_cases(self):
        tracer = HarnessTracer()

        async def run_case(index):
            with tracer.trace(EVAL_CASE_SPAN) as eval_case_span:
                with tracer.trace("parent") as parent:
                    await asyncio.sleep(0)

                    def worker():
                        with tracer.trace("haystack.chat_generator.run", parent_span=parent) as span:
                            span.set_content_tag(
                                "haystack.component.output",
                                {
                                    "replies": [
                                        ChatMessage.from_assistant(
                                            "answer",
                                            meta={
                                                "model": str(index),
                                                "usage": {"input_tokens": index, "output_tokens": 1},
                                            },
                                        )
                                    ]
                                },
                            )

                    await asyncio.to_thread(worker)
            return eval_case_span.collected.summarize()

        async def run_all():
            return await asyncio.gather(*(run_case(i) for i in range(1, 5)))

        with tracer.activate():
            results = asyncio.run(run_all())
        for index, summary in enumerate(results, start=1):
            assert list(summary.models) == [str(index)]
            assert summary.models[str(index)].input_tokens == index

    def test_reply_without_usage_meta(self):
        tracer = HarnessTracer()
        with tracer.trace(EVAL_CASE_SPAN) as eval_case_span:
            with tracer.trace("haystack.chat_generator.run") as span:
                span.set_content_tag("haystack.component.output", {"replies": [ChatMessage.from_assistant("hi")]})
        assert not eval_case_span.collected.summarize().all_tokens_reported

    def test_activate_restores_tracing(self):
        tracer = HarnessTracer()
        try:
            with tracer.activate():
                msg = "evaluation failed"
                raise RuntimeError(msg)
        except RuntimeError:
            pass
        assert tracing.tracer.actual_tracer is not tracer

    def test_records_carry_parent_ids(self):
        """A record built from a stored trace has parent pointers, so a live one carries them too."""
        tracer = HarnessTracer()
        with tracer.trace(EVAL_CASE_SPAN) as eval_case_span:
            with tracer.trace("haystack.component.run", tags={"haystack.component.name": "ranker"}):
                with tracer.trace("haystack.chat_generator.run"):
                    pass
        spans = eval_case_span.collected.spans
        by_id = {span.span_id: span for span in spans}
        # Spans arrive in the order they ended, so the nested generator comes first.
        generator, ranker = spans
        assert by_id[generator.parent_span_id] is ranker
        assert ranker.parent_span_id == eval_case_span.span_id


class TestMeasureOutput:
    def test_samples_string_sockets(self):
        """A count says how many queries were issued; only the text says whether they decomposed or restated."""
        sizes, texts = _measure_output(
            value={
                "queries": ["who owns it", "when was it sold"],
                "prompt": "one rendered prompt",
                "documents": [Document(content="a long article body")],
                "things": ["a string", 7],
            }
        )
        assert sizes == {"queries": 2, "prompt": 1, "documents": 1, "things": 2}
        # Only a socket that is nothing but strings is sampled, so document text is never retained.
        assert texts == {"queries": ["who owns it", "when was it sold"], "prompt": ["one rendered prompt"]}

    def test_caps_samples(self):
        long_query = "x" * (MAX_RECORDED_TEXT_CHARS + 50)
        sizes, texts = _measure_output(value={"queries": [long_query] * (MAX_RECORDED_TEXTS + 20)})
        assert texts["queries"] == [f"{'x' * MAX_RECORDED_TEXT_CHARS}..."] * MAX_RECORDED_TEXTS
        # The count is not capped, so the sample being short never hides how much was really emitted.
        assert sizes["queries"] == MAX_RECORDED_TEXTS + 20


class TestReportedTokens:
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            pytest.param(
                {"input_tokens": 3, "output_tokens": 1},
                ModelTokenUsage(input_tokens=3, output_tokens=1),
                id="named_counts",
            ),
            pytest.param(
                {"prompt_tokens": 3, "completion_tokens": 1},
                ModelTokenUsage(input_tokens=3, output_tokens=1),
                id="provider_keys",
            ),
            pytest.param({"input_tokens": 3}, None, id="output_count_missing"),
            pytest.param({"input_tokens": 3, "output_tokens": "lots"}, None, id="count_is_not_a_number"),
            pytest.param({}, None, id="nothing_reported"),
            pytest.param(None, None, id="no_usage_key"),
        ],
    )
    def test_reported_tokens(self, usage, expected):
        """A generator reports whatever its provider does, so the counts are normalized where they arrive."""
        assert _reported_tokens(usage=usage) == expected


class TestCollectedSpans:
    def test_sums_repeated_models(self):
        """Summarizing is a pure function of the collected spans, so it can be checked without running anything."""
        summary = CollectedSpans(spans=[generator_span(), generator_span()]).summarize()
        assert summary.llm_calls == 2
        assert summary.models["m"] == ModelTokenUsage(input_tokens=6, output_tokens=2)
        assert summary.all_tokens_reported

    def test_call_without_output(self):
        """A generator span that never emitted an output tag spent tokens nobody can account for."""
        silent = HarnessSpan(is_generator_span=True)
        summary = CollectedSpans(spans=[generator_span(), silent]).summarize()
        assert summary.all_tokens_reported is False
        # The silent call is not counted, because nothing says it produced anything.
        assert summary.llm_calls == 1

    def test_unquantified_usage(self):
        for span in (generator_span(tokens=None), generator_span(model=None)):
            summary = CollectedSpans(spans=[span]).summarize()
            assert summary.all_tokens_reported is False
            assert summary.models == {}
            # The call still happened, so it is still counted.
            assert summary.llm_calls == 1

    def test_generator_output_not_a_stage(self):
        """A reply count says nothing about how much reached the next stage, and would collide with its owner."""
        ranker = HarnessSpan(component_name="ranker")
        ranker.output_sizes = {"documents": 4}
        nested = generator_span(component_name="ranker")
        nested.output_sizes = {"replies": 1}
        summary = CollectedSpans(spans=[nested, ranker]).summarize()
        assert summary.outputs == {"ranker": {"documents": 4}}
