import contextlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from haystack import tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer

from haystack_integrations.agent_pack.optimization import (
    LocalTraceCollector,
    LocalTraceStore,
    TraceArtifact,
    TraceCapturingAgentRunner,
    TraceSelection,
    extract_agent_reference_output,
    extract_agent_replay_inputs,
)


class RecordingSpan(Span):
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.tags = {}

    def set_tag(self, key, value):
        self.tags[key] = value


class RecordingTracer(Tracer):
    def __init__(self):
        self.spans = []
        self.active = []

    @contextlib.contextmanager
    def trace(self, operation_name, tags=None, parent_span=None):
        assert parent_span is None or isinstance(parent_span, RecordingSpan)
        span = RecordingSpan(operation_name)
        span.tags.update(tags or {})
        self.spans.append(span)
        self.active.append(span)
        try:
            yield span
        finally:
            self.active.pop()

    def current_span(self):
        return self.active[-1] if self.active else None


@pytest.fixture(autouse=True)
def restore_tracing():
    previous = tracing.tracer.actual_tracer
    content = tracing.tracer.is_content_tracing_enabled
    yield
    tracing.enable_tracing(previous)
    tracing.tracer.is_content_tracing_enabled = content


def test_capture_run_emits_platform_shape_hierarchy_and_delegates():
    delegate = RecordingTracer()
    tracing.enable_tracing(delegate)
    collector = LocalTraceCollector()

    with collector.capture_run() as capture:
        with tracing.tracer.trace("root", tags={"structured": {"items": [1, 2]}}) as root:
            root.set_content_tag("content", {"question": "hello"})
            with tracing.tracer.trace("child", parent_span=root):
                pass

    artifact = capture.to_artifact()
    assert set(artifact.to_dict()) == {
        "schema_version",
        "run_id",
        "started_at",
        "finished_at",
        "duration_ms",
        "status",
        "traces",
        "logs",
        "failure",
    }
    assert artifact.schema_version == "haystack-trace/v1"
    assert artifact.logs == ()
    assert artifact.traces[0]["tags"]["structured"] == {"items": [1, 2]}
    assert artifact.traces[0]["tags"]["content"] == {"question": "hello"}
    assert artifact.traces[1]["parent_span_id"] == artifact.traces[0]["span_id"]
    assert [span.operation_name for span in delegate.spans] == ["root", "child"]
    assert tracing.tracer.actual_tracer is delegate


def test_capture_failure_is_stored_and_reraised():
    store = LocalTraceStore()
    collector = LocalTraceCollector(store)

    with pytest.raises(RuntimeError, match="boom"):
        with collector.capture_run():
            with tracing.tracer.trace("failing"):
                message = "boom"
                raise RuntimeError(message)

    artifact = store.list(TraceSelection(status="failed"))[0]
    assert artifact.status == "failed"
    assert artifact.failure["type"] == "RuntimeError"
    assert artifact.traces[0]["tags"]["error"] is True


def test_local_store_persists_and_selects(tmp_path):
    store = LocalTraceStore(tmp_path)
    collector = LocalTraceCollector(store)
    for operation in ("one", "two"):
        with collector.capture_run():
            with tracing.tracer.trace(operation):
                pass

    restored = LocalTraceStore(tmp_path)
    selected = restored.list(TraceSelection(run_ids=frozenset({store.list()[0].run_id}), limit=1))
    assert len(selected) == 1
    assert TraceArtifact.from_dict(selected[0].to_dict()) == selected[0]


def test_concurrent_captures_are_isolated():
    collector = LocalTraceCollector()

    def capture(index):
        with collector.capture_run() as run:
            with tracing.tracer.trace(f"run-{index}"):
                pass
        return run.to_artifact()

    with ThreadPoolExecutor(max_workers=4) as executor:
        artifacts = list(executor.map(capture, range(8)))

    assert len({artifact.run_id for artifact in artifacts}) == 8
    assert [{span["operation_name"] for span in artifact.traces} for artifact in artifacts] == [
        {f"run-{index}"} for index in range(8)
    ]


def test_agent_runner_captures_replayable_content():
    agent = Agent(chat_generator=MockChatGenerator([ChatMessage.from_assistant("answer")]))
    run = TraceCapturingAgentRunner().run(agent, messages=[ChatMessage.from_user("question")])

    inputs = extract_agent_replay_inputs(run.trace)
    output = extract_agent_reference_output(run.trace)
    assert ChatMessage.from_dict(inputs["messages"][0]).text == "question"
    assert ChatMessage.from_dict(output["last_message"]).text == "answer"


def test_replay_extraction_requires_content_tracing():
    collector = LocalTraceCollector(content_tracing=False)
    with collector.capture_run() as capture:
        with tracing.tracer.trace("haystack.agent.run"):
            pass

    with pytest.raises(ValueError, match="content tracing enabled"):
        extract_agent_replay_inputs(capture.to_artifact())


@pytest.mark.asyncio
async def test_agent_runner_supports_async_runs():
    agent = Agent(chat_generator=MockChatGenerator([ChatMessage.from_assistant("answer")]))
    run = await TraceCapturingAgentRunner().run_async(agent, messages=[ChatMessage.from_user("question")])
    assert run.result["last_message"].text == "answer"
    assert run.trace.status == "success"
