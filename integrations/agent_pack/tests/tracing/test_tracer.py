import asyncio

from haystack import Document, Pipeline, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.dataclasses import ChatMessage

from haystack_integrations.evaluation.dataclasses import ModelTokenUsage
from haystack_integrations.tracing.agent_pack import (
    EVAL_CASE_SPAN,
    HarnessTracer,
    ReportedUsage,
    SpanRecord,
    eval_case_usage_from_records,
    usage_from_span,
)
from haystack_integrations.tracing.agent_pack.span_records import MAX_RECORDED_TEXT_CHARS, MAX_RECORDED_TEXTS


def test_ranker_and_agent_usage_counted_once_without_content_tracing():
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
        usage = usage_from_span(span=span)
        assert usage.complete
        assert usage.calls == 2
        assert usage.models["ranker"].input_tokens == 7
        assert usage.models["coordinator"].input_tokens == 11
    finally:
        tracing.tracer.is_content_tracing_enabled = old_content


def test_concurrent_eval_cases_and_threaded_parents_keep_usage_separate():
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
        return usage_from_span(span=eval_case_span)

    async def run_all():
        return await asyncio.gather(*(run_case(i) for i in range(1, 5)))

    with tracer.activate():
        results = asyncio.run(run_all())
    for index, usage in enumerate(results, start=1):
        assert list(usage.models) == [str(index)]
        assert usage.models[str(index)].input_tokens == index


def test_missing_usage_is_unavailable_and_tracer_is_disabled_after_failure():
    tracer = HarnessTracer()
    try:
        with tracer.activate(), tracing.tracer.trace(EVAL_CASE_SPAN) as eval_case_span:
            with tracer.trace("haystack.chat_generator.run") as span:
                span.set_content_tag("haystack.component.output", {"replies": [ChatMessage.from_assistant("no usage")]})
            msg = "evaluation failed"
            raise RuntimeError(msg)
    except RuntimeError:
        pass
    assert not usage_from_span(span=eval_case_span).complete
    assert tracing.tracer.actual_tracer is not tracer


def emit(tracer, component, output):
    """Emit one component's output the way Haystack's component tracing does."""
    with tracer.trace("haystack.component.run", tags={"haystack.component.name": component}) as span:
        span.set_content_tag("haystack.component.output", output)


def test_a_stage_that_rewrites_the_question_records_what_it_asked():
    """A count says four queries were issued; only the text says whether they decomposed or restated."""
    tracer = HarnessTracer()

    with tracer.trace(EVAL_CASE_SPAN) as span:
        emit(tracer, "expander", {"queries": ["who owns it", "when was it sold"]})
    usage = usage_from_span(span=span)

    assert usage.outputs["expander"] == {"queries": 2}
    assert usage.texts["expander"] == {"queries": ["who owns it", "when was it sold"]}


def test_an_unbounded_expansion_cannot_fill_the_readers_context():
    tracer = HarnessTracer()
    long_query = "x" * (MAX_RECORDED_TEXT_CHARS + 50)

    with tracer.trace(EVAL_CASE_SPAN) as span:
        emit(tracer, "expander", {"queries": [long_query] * (MAX_RECORDED_TEXTS + 20)})
    usage = usage_from_span(span=span)

    kept = usage.texts["expander"]["queries"]
    assert len(kept) == MAX_RECORDED_TEXTS
    assert all(entry == "x" * MAX_RECORDED_TEXT_CHARS + "..." for entry in kept)
    # The count is not capped, so the sample being short never hides how much was really emitted.
    assert usage.outputs["expander"]["queries"] == MAX_RECORDED_TEXTS + 20


def test_documents_are_counted_and_never_sampled():
    """Sampling is for sockets that are nothing but short strings, so document text is not retained."""
    tracer = HarnessTracer()

    with tracer.trace(EVAL_CASE_SPAN) as span:
        emit(tracer, "retriever", {"documents": [Document(content="a long article body")]})
        emit(tracer, "mixed", {"things": ["a string", 7]})
    usage = usage_from_span(span=span)

    assert usage.outputs == {"retriever": {"documents": 1}, "mixed": {"things": 2}}
    assert usage.texts == {}


def generator_record(model="m", tokens=None, **overrides):
    """One span record shaped like a model call that reported a single reply."""
    return SpanRecord(
        is_generator_span=True,
        reported_output=True,
        reported_usage=[
            ReportedUsage(model=model, tokens=tokens if tokens is not None else {"input_tokens": 3, "output_tokens": 1})
        ],
        **overrides,
    )


def test_the_same_model_called_twice_is_summed():
    """Folding is a pure function of the records, so it can be checked without running anything."""
    usage = eval_case_usage_from_records(records=[generator_record(), generator_record()])

    assert usage.calls == 2
    assert usage.models["m"] == ModelTokenUsage(input_tokens=6, output_tokens=2)
    assert usage.complete


def test_a_model_call_that_reported_nothing_makes_the_measurement_unpriceable():
    """A generator span that never emitted an output tag spent tokens nobody can account for."""
    silent = SpanRecord(is_generator_span=True)

    usage = eval_case_usage_from_records(records=[generator_record(), silent])

    assert usage.complete is False
    # The silent call is not counted, because nothing says it produced anything.
    assert usage.calls == 1


def test_usage_a_generator_declined_to_quantify_is_not_guessed_at():
    replied_without_usage = generator_record(model="m", tokens={})
    replied_without_model = generator_record(model=None)

    for record in (replied_without_usage, replied_without_model):
        usage = eval_case_usage_from_records(records=[record])
        assert usage.complete is False
        assert usage.models == {}
        # The call still happened, so it is still counted.
        assert usage.calls == 1


def test_a_generators_own_output_is_not_reported_as_a_stage():
    """A reply count says nothing about how much reached the next stage, and would collide with its owner."""
    ranker = SpanRecord(component_name="ranker", output_sizes={"documents": 4})
    nested = generator_record(component_name="ranker", output_sizes={"replies": 1})

    usage = eval_case_usage_from_records(records=[nested, ranker])

    assert usage.outputs == {"ranker": {"documents": 4}}


def test_records_carry_the_nesting_a_stored_trace_would_have():
    """A record built from a stored trace has parent pointers, so a live one carries them too."""
    tracer = HarnessTracer()

    with tracer.trace(EVAL_CASE_SPAN) as eval_case_span:
        with tracer.trace("haystack.component.run", tags={"haystack.component.name": "ranker"}):
            with tracer.trace("haystack.chat_generator.run"):
                pass

    records = eval_case_span.collected.records
    by_id = {record.span_id: record for record in records}
    # Records arrive in the order their spans ended, so the nested generator comes first.
    generator, ranker = records
    assert by_id[generator.parent_span_id] is ranker
    assert ranker.parent_span_id == eval_case_span.record.span_id
    assert generator.is_generator_span and ranker.component_name == "ranker"
