import asyncio

from haystack import Document, Pipeline, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tracing.agent_pack.tracer import UsageTracer


def test_ranker_and_agent_usage_counted_once_without_content_tracing():
    tracer = UsageTracer()
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
        with tracer.activate(), tracer.case() as usage:
            pipeline.run({"ranker": {"query": "Berlin", "documents": [Document(content="Berlin")]}})
            agent.run(messages=[ChatMessage.from_user("q")])
        assert usage.complete
        assert usage.calls == 2
        assert usage.models["ranker"].input_tokens == 7
        assert usage.models["coordinator"].input_tokens == 11
    finally:
        tracing.tracer.is_content_tracing_enabled = old_content


def test_concurrent_cases_and_threaded_parents_keep_usage_separate():
    tracer = UsageTracer()

    async def run_case(index):
        with tracer.case() as usage:
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
        return usage

    async def run_all():
        return await asyncio.gather(*(run_case(i) for i in range(1, 5)))

    with tracer.activate():
        results = asyncio.run(run_all())
    for index, usage in enumerate(results, start=1):
        assert list(usage.models) == [str(index)]
        assert usage.models[str(index)].input_tokens == index


def test_missing_usage_is_unavailable_and_tracer_is_disabled_after_failure():
    tracer = UsageTracer()
    try:
        with tracer.activate(), tracer.case() as usage:
            with tracer.trace("haystack.chat_generator.run") as span:
                span.set_content_tag("haystack.component.output", {"replies": [ChatMessage.from_assistant("no usage")]})
            msg = "evaluation failed"
            raise RuntimeError(msg)
    except RuntimeError:
        pass
    assert not usage.complete
    assert tracing.tracer.actual_tracer is not tracer
