import contextlib
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from haystack import Document, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.tracing import Span, Tracer

from haystack_integrations.agent_pack.optimization import (
    LocalTraceCollector,
    LocalTraceStore,
    TraceArtifact,
    TraceCaptureLimits,
    TraceCapturingAgentRunner,
    TraceSelection,
    span_tag,
)
from haystack_integrations.agent_pack.optimization.tracing import (
    extract_agent_reference_output,
    extract_agent_replay_inputs,
    is_replayable,
)


class RecordingSpan(Span):
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.tags = {}
        self.underlying = object()

    def set_tag(self, key, value):
        self.tags[key] = value

    def raw_span(self):
        return self.underlying


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


def echo_tool(value: str) -> str:
    return value


def _tool(name):
    return Tool(
        name=name,
        description=f"Echo for {name}.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        function=echo_tool,
    )


def test_capture_run_emits_expected_shape_hierarchy_and_delegates():
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
    assert span_tag(span=artifact.traces[0], key="structured") == {"items": [1, 2]}
    assert span_tag(span=artifact.traces[0], key="content") == {"question": "hello"}
    assert artifact.traces[1]["parent_span_id"] == artifact.traces[0]["span_id"]
    assert [span.operation_name for span in delegate.spans] == ["root", "child"]
    assert tracing.tracer.actual_tracer is delegate


def test_capture_does_not_leak_content_to_the_delegate_tracer():
    """Capturing content locally must not start exporting prompts and documents to an installed tracer."""
    delegate = RecordingTracer()
    tracing.enable_tracing(delegate)
    tracing.tracer.is_content_tracing_enabled = False

    with LocalTraceCollector().capture_run() as capture:
        with tracing.tracer.trace("root") as root:
            root.set_content_tag("secret", {"question": "hello"})
            root.set_tag("public", 1)

    assert span_tag(span=capture.to_artifact().traces[0], key="secret") == {"question": "hello"}
    assert "secret" not in delegate.spans[0].tags
    assert delegate.spans[0].tags["public"] == 1
    assert tracing.tracer.is_content_tracing_enabled is False


def test_content_still_reaches_the_delegate_when_the_process_enabled_it():
    delegate = RecordingTracer()
    tracing.enable_tracing(delegate)
    tracing.tracer.is_content_tracing_enabled = True

    with LocalTraceCollector().capture_run():
        with tracing.tracer.trace("root") as root:
            root.set_content_tag("shared", "value")

    assert delegate.spans[0].tags["shared"] == "value"


def test_capture_content_can_be_disabled_locally():
    with LocalTraceCollector(capture_content=False).capture_run() as capture:
        with tracing.tracer.trace("root") as root:
            root.set_content_tag("secret", "value")
            root.set_tag("public", "value")

    span = capture.to_artifact().traces[0]
    assert "secret" not in span["tags"]
    assert span_tag(span=span, key="public") == "value"


def test_combined_span_exposes_the_delegate_raw_span():
    delegate = RecordingTracer()
    tracing.enable_tracing(delegate)

    with LocalTraceCollector().capture_run():
        with tracing.tracer.trace("root") as root:
            assert root.raw_span() is delegate.spans[0].underlying
            assert tracing.tracer.current_span() is root


def test_capture_limits_drop_embeddings_and_truncate_oversized_values():
    limits = TraceCaptureLimits(max_string_length=8, max_sequence_items=2, dropped_keys=("embedding",))
    with LocalTraceCollector(limits=limits).capture_run() as capture:
        with tracing.tracer.trace("root") as root:
            root.set_content_tag("documents", [Document(content="a" * 20, embedding=[0.1, 0.2, 0.3])])
            root.set_tag("items", [1, 2, 3, 4])

    span = capture.to_artifact().traces[0]
    documents = span_tag(span=span, key="documents")
    # Captured tags deserialize back into the objects they were recorded from.
    assert isinstance(documents[0], Document)
    assert documents[0].embedding is None
    assert documents[0].content == "aaaaaaaa <truncated: 20 chars total>"
    assert span_tag(span=span, key="items") == [1, 2]
    assert span["tags"]["items"]["serialization_schema"]["truncatedItems"] == 2


def test_capture_failure_is_stored_and_reraised():
    store = LocalTraceStore()
    collector = LocalTraceCollector(store=store)

    with pytest.raises(RuntimeError, match="boom"):
        with collector.capture_run():
            with tracing.tracer.trace("failing"):
                message = "boom"
                raise RuntimeError(message)

    artifact = store.list(TraceSelection(status="failed"))[0]
    assert artifact.status == "failed"
    assert artifact.failure["type"] == "RuntimeError"
    assert span_tag(span=artifact.traces[0], key="error") is True


def test_local_store_persists_and_selects(tmp_path):
    store = LocalTraceStore(tmp_path)
    collector = LocalTraceCollector(store=store)
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


def test_concurrent_tool_calls_are_captured_and_parented():
    """Haystack runs tool calls concurrently; every tool span must still land under its step span."""
    agent = Agent(
        chat_generator=MockChatGenerator(
            [
                ChatMessage.from_assistant(
                    tool_calls=[
                        ToolCall("first", {"value": "a"}, id="1"),
                        ToolCall("second", {"value": "b"}, id="2"),
                    ]
                ),
                ChatMessage.from_assistant("done"),
            ]
        ),
        tools=[_tool("first"), _tool("second")],
        tool_concurrency_limit=2,
    )
    run = TraceCapturingAgentRunner().run(agent, messages=[ChatMessage.from_user("go")])

    spans = {span["span_id"]: span for span in run.trace.traces}
    tool_spans = [span for span in run.trace.traces if span["operation_name"] == "haystack.agent.step.tool"]
    step_ids = {span["span_id"] for span in run.trace.traces if span["operation_name"] == "haystack.agent.step"}
    assert len(tool_spans) == 2
    assert {span_tag(span=span, key="haystack.tool.name") for span in tool_spans} == {"first", "second"}
    assert all(span["parent_span_id"] in step_ids for span in tool_spans)
    assert all(spans[span["parent_span_id"]]["operation_name"] == "haystack.agent.step" for span in tool_spans)


def test_agent_runner_captures_replayable_content():
    agent = Agent(chat_generator=MockChatGenerator([ChatMessage.from_assistant("answer")]))
    run = TraceCapturingAgentRunner().run(agent, messages=[ChatMessage.from_user("question")])

    inputs = extract_agent_replay_inputs(artifact=run.trace)
    output = extract_agent_reference_output(artifact=run.trace)
    # Captured content round-trips through Haystack's schema-aware serialization, so replay inputs come back as the
    # objects they were recorded from rather than as raw dictionaries.
    assert isinstance(inputs["messages"][0], ChatMessage)
    assert inputs["messages"][0].text == "question"
    assert output["last_message"].text == "answer"
    assert is_replayable(artifact=run.trace)


def test_replay_extraction_finds_a_nested_agent_span():
    """An Agent run inside surrounding instrumentation is still replayable."""
    with LocalTraceCollector().capture_run() as capture:
        with tracing.tracer.trace("surrounding.pipeline") as outer:
            with tracing.tracer.trace("haystack.agent.run", parent_span=outer) as agent_span:
                agent_span.set_content_tag("haystack.agent.input", {"messages": []})

    assert extract_agent_replay_inputs(artifact=capture.to_artifact()) == {"messages": []}


def test_replay_extraction_reports_missing_agent_span_and_missing_content():
    with LocalTraceCollector().capture_run() as without_agent:
        with tracing.tracer.trace("something.else"):
            pass
    with pytest.raises(ValueError, match=re.escape("no haystack.agent.run span")):
        extract_agent_replay_inputs(artifact=without_agent.to_artifact())
    assert not is_replayable(artifact=without_agent.to_artifact())

    with LocalTraceCollector(capture_content=False).capture_run() as without_content:
        with tracing.tracer.trace("haystack.agent.run"):
            pass
    with pytest.raises(ValueError, match="capture_content=True"):
        extract_agent_replay_inputs(artifact=without_content.to_artifact())


def test_a_second_collector_cannot_install_concurrently():
    first = LocalTraceCollector()
    with first.install():
        with pytest.raises(RuntimeError, match="already installed"):
            with LocalTraceCollector().install():
                pass


@pytest.mark.asyncio
async def test_agent_runner_supports_async_runs():
    agent = Agent(chat_generator=MockChatGenerator([ChatMessage.from_assistant("answer")]))
    run = await TraceCapturingAgentRunner().run_async(agent, messages=[ChatMessage.from_user("question")])
    assert run.result["last_message"].text == "answer"
    assert run.trace.status == "success"
