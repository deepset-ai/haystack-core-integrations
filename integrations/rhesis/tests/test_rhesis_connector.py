# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rhesis.telemetry.constants import ConversationContext, TestExecutionContext

from haystack_integrations.components.connectors.rhesis import RhesisConnector
from haystack_integrations.tracing.rhesis import DefaultSpanHandler, rhesis_invocation_context
from haystack_integrations.tracing.rhesis.tracer import tracing_context_var

_PROVIDER_PATH = "haystack_integrations.components.connectors.rhesis.rhesis_connector.build_tracer_provider"


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span, component_type=None):
        pass


class TestRhesisConnector:
    def test_run(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch(_PROVIDER_PATH):
            connector = RhesisConnector(name="Chat example")
            mock_tracer = Mock()
            mock_tracer.get_trace_url.return_value = "http://localhost:3000/traces?open_trace=abc"
            mock_tracer.get_trace_id.return_value = "abc123"
            connector.tracer = mock_tracer

            response = connector.run(invocation_context={"session_id": "sess-1"})
            assert response["name"] == "Chat example"
            assert response["trace_url"] == "http://localhost:3000/traces?open_trace=abc"
            assert response["trace_id"] == "abc123"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch(_PROVIDER_PATH):
            connector = RhesisConnector(name="Chat example")
            serialized = connector.to_dict()

        assert serialized == {
            "type": "haystack_integrations.components.connectors.rhesis.rhesis_connector.RhesisConnector",
            "init_parameters": {
                "name": "Chat example",
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["RHESIS_API_KEY"],
                    "strict": True,
                },
                "base_url": None,
                "project_id": None,
                "environment": None,
                "frontend_url": None,
                "span_handler": None,
            },
        }

    def test_to_dict_does_not_bake_in_this_machines_environment(self, monkeypatch):
        """
        A pipeline dumped where `RHESIS_BASE_URL` points at a laptop must not carry that to prod.

        Anything the caller left to the environment stays `None` in the serialized form, so
        deserializing resolves it wherever the pipeline actually runs.
        """
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        monkeypatch.setenv("RHESIS_BASE_URL", "http://localhost:8080")
        monkeypatch.setenv("RHESIS_ENVIRONMENT", "laptop")
        monkeypatch.setenv("RHESIS_PROJECT_ID", "local-project")
        with patch(_PROVIDER_PATH):
            connector = RhesisConnector(name="Chat example")
            serialized = connector.to_dict()["init_parameters"]

        # Resolved for this process...
        assert connector.base_url == "http://localhost:8080"
        assert connector.environment == "laptop"
        assert connector.project_id == "local-project"
        # ...but not written into the pipeline definition.
        assert serialized["base_url"] is None
        assert serialized["environment"] is None
        assert serialized["project_id"] is None

    def test_to_dict_keeps_explicit_arguments(self, monkeypatch):
        """What the caller passed explicitly is theirs, and must survive the round trip."""
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        monkeypatch.setenv("RHESIS_BASE_URL", "http://localhost:8080")
        with patch(_PROVIDER_PATH):
            connector = RhesisConnector(name="Chat example", base_url="https://api.rhesis.ai", environment="prod")
            serialized = connector.to_dict()["init_parameters"]

        assert serialized["base_url"] == "https://api.rhesis.ai"
        assert serialized["environment"] == "prod"

    def test_to_dict_with_custom_handler(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch(_PROVIDER_PATH):
            connector = RhesisConnector(name="Chat example", span_handler=CustomSpanHandler())
            serialized = connector.to_dict()

        assert serialized["init_parameters"]["span_handler"]["type"].endswith("CustomSpanHandler")

    def test_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch(_PROVIDER_PATH):
            connector = RhesisConnector(
                name="Chat example",
                base_url="http://localhost:8080",
                project_id="proj",
                environment="staging",
                frontend_url="http://localhost:3000",
            )
            data = connector.to_dict()
            restored = RhesisConnector.from_dict(data)
            assert restored.name == connector.name
            assert restored.base_url == connector.base_url
            assert restored.project_id == connector.project_id
            assert restored.environment == connector.environment
            assert restored.frontend_url == connector.frontend_url

    def test_unset_api_key_env_var_raises(self, monkeypatch):
        """`Secret.from_env_var` is strict, so Haystack rejects it before the connector looks."""
        monkeypatch.delenv("RHESIS_API_KEY", raising=False)
        with (
            patch(_PROVIDER_PATH),
            pytest.raises(ValueError, match="authentication environment variables are set"),
        ):
            RhesisConnector(name="Chat example", api_key=Secret.from_env_var("RHESIS_API_KEY"))

    def test_api_key_none_raises(self):
        """Passing `api_key=None` explicitly is the path the connector's own check exists for."""
        with (
            patch(_PROVIDER_PATH),
            pytest.raises(ValueError, match="RHESIS_API_KEY is required"),
        ):
            RhesisConnector(name="Chat example", api_key=None)

    def test_pipeline_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch(_PROVIDER_PATH):
            pipe = Pipeline()
            pipe.add_component("tracer", RhesisConnector("Chat example"))
            pipe.add_component("prompt_builder", ChatPromptBuilder())
            yaml = pipe.dumps()
            restored = Pipeline.loads(yaml)
            tracer = restored.get_component("tracer")
            assert tracer.name == "Chat example"
            assert "token-value" not in yaml
            assert "test-key" not in yaml

    def test_enable_tracing_called(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with (
            patch(_PROVIDER_PATH),
            patch(
                "haystack_integrations.components.connectors.rhesis.rhesis_connector.tracing.enable_tracing"
            ) as mock_enable,
        ):
            RhesisConnector(name="Chat example")
            mock_enable.assert_called_once()


class TestProviderOwnership:
    """The connector owns its OTel provider instead of claiming the process-wide one."""

    @staticmethod
    def _connector(**kwargs):
        """A connector on the real ``build_tracer_provider`` path, with its provider disposed after."""
        return RhesisConnector(api_key=Secret.from_token("test-key"), **kwargs)

    def test_construction_leaves_the_otel_global_alone(self):
        """
        A Haystack user with their own APM must keep it.

        ``get_tracer_provider`` called ``trace.set_tracer_provider``, so whichever of Rhesis and the
        host's instrumentation initialised first won the global and OpenTelemetry refused the other's
        override — silently sending the host's spans here, or losing ours to them. Nothing in this
        integration needs the global: spans go through ``telemetry.otel_tracer``.
        """
        before = trace.get_tracer_provider()

        connector = self._connector(name="Chat example")
        try:
            assert trace.get_tracer_provider() is before
            assert connector.tracer.telemetry.provider is not trace.get_tracer_provider()
        finally:
            connector.tracer.telemetry.provider.shutdown()

    def test_each_connector_gets_its_own_provider(self):
        """
        Two connectors in one process must route to their own projects.

        ``get_tracer_provider`` cached the first provider it built, and the exporter that stamps
        ``project_id`` and holds the endpoint and API key is created with it — so a second connector's
        project, base URL and key were discarded and its spans went to the first one's project.
        """
        first = self._connector(name="first", project_id="project-AAA")
        second = self._connector(name="second", project_id="project-BBB")
        try:
            assert first.tracer.telemetry.provider is not second.tracer.telemetry.provider
            assert first.tracer.telemetry.project_id == "project-AAA"
            assert second.tracer.telemetry.project_id == "project-BBB"
        finally:
            first.tracer.telemetry.provider.shutdown()
            second.tracer.telemetry.provider.shutdown()

    def test_invocation_context_outside_a_traced_run_is_ignored(self, caplog):
        """
        ``run()`` called with no root span open has nowhere to put the context, so it drops it.

        The ContextVar is scoped by ``RhesisTracer.trace``, which sets a restore point when the root
        span opens. Written outside that scope there is no restore point, and the value would become
        the default for every later run in the process that supplies none.
        """
        with patch(_PROVIDER_PATH):
            connector = self._connector(name="Chat example")

        connector.run(invocation_context={"session_id": "alice"})

        assert tracing_context_var.get({}) == {}
        assert "outside a traced run" in caplog.text


@component
class _Echo:
    """Chat-shaped component, so the pipeline root can extract conversation text."""

    @component.output_types(replies=list)
    def run(self, messages: list[ChatMessage]) -> dict:
        return {"replies": [ChatMessage.from_assistant("ok")]}


@component
class _Upper:
    """No chat messages anywhere, so no conversation text is extractable."""

    @component.output_types(text=str)
    def run(self, text: str) -> dict:
        return {"text": text.upper()}


class TestInvocationContext:
    """The ``invocation_context`` input socket, end to end through a real pipeline."""

    @staticmethod
    def _traced_pipeline(worker, name):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with patch(_PROVIDER_PATH, return_value=provider):
            connector = RhesisConnector(name="test", api_key=Secret.from_token("test-key"))
        pipe = Pipeline()
        pipe.add_component("tracer", connector)
        pipe.add_component(name, worker)
        return pipe, exporter

    @staticmethod
    def _root_attributes(exporter):
        roots = [span for span in exporter.get_finished_spans() if span.parent is None]
        assert len(roots) == 1, f"expected exactly one root span, got {len(roots)}"
        return dict(roots[0].attributes)

    def test_context_does_not_leak_into_the_next_run(self):
        """
        A run that passes no ``invocation_context`` must not inherit the previous run's.

        ``run()`` sets a ContextVar from inside the pipeline and has no teardown hook of its own, so
        before the tracer scoped it to the root span, run N+1 was stamped with run N's session — one
        user's turn filed under another user's conversation.
        """
        pipe, exporter = self._traced_pipeline(_Echo(), "echo")
        payload = {"echo": {"messages": [ChatMessage.from_user("hi")]}}

        pipe.run({**payload, "tracer": {"invocation_context": {"session_id": "alice"}}})
        first = self._root_attributes(exporter)

        exporter.clear()
        pipe.run(payload)
        second = self._root_attributes(exporter)

        assert first[ConversationContext.SpanAttributes.CONVERSATION_ID] == "alice"
        assert ConversationContext.SpanAttributes.CONVERSATION_ID not in second

    def test_content_flag_off_keeps_the_message_off_rhesis_attributes(self, monkeypatch):
        """
        With content tracing off, the user's message must not reach `rhesis.conversation.*`.

        `handle()` promotes that text out of `span.get_data()`, which `set_tag` fills whether the
        flag is set or not, so the promotion has to check the flag itself. The README tells users
        this flag controls exactly this.

        `haystack.pipeline.input_data` is deliberately not asserted absent: Haystack hands pipeline
        I/O over as an ordinary tag rather than a content tag, so it is stamped regardless — the same
        behaviour as the langfuse integration. README "Limitations" records it.
        """
        monkeypatch.setattr("haystack.tracing.tracer.is_content_tracing_enabled", False)
        secret = "SENSITIVE-USER-QUERY-12345"

        pipe, exporter = self._traced_pipeline(_Echo(), "echo")
        pipe.run({"echo": {"messages": [ChatMessage.from_user(secret)]}})

        attrs = ConversationContext.SpanAttributes
        for span in exporter.get_finished_spans():
            for key in (attrs.CONVERSATION_INPUT, attrs.CONVERSATION_OUTPUT):
                assert secret not in str(span.attributes.get(key, ""))
            for event in span.events:
                assert secret not in str(dict(event.attributes or {}))

    def test_scoped_context_reaches_every_span(self):
        """
        Filtering a trace's child spans by your own run id is the point of passing one.

        The mapped attributes used to land on the root span only, so that query returned nothing.
        `rhesis_invocation_context` is the path that can guarantee full coverage: it is set before
        the run starts, so no span opens without it.

        The turn-root flag stays on the root alone — on a child the exporter would strip its real
        parent and detach the subtree into a turn of its own.
        """
        pipe, exporter = self._traced_pipeline(_Echo(), "echo")
        with rhesis_invocation_context({"session_id": "dave", "test_run_id": "tr-9", "user_id": "u-1"}):
            pipe.run({"echo": {"messages": [ChatMessage.from_user("hi")]}})

        attrs = ConversationContext.SpanAttributes
        spans = exporter.get_finished_spans()
        children = [span for span in spans if span.parent is not None]
        assert children, "expected at least one child span"

        for span in spans:
            assert span.attributes[attrs.CONVERSATION_ID] == "dave"
            assert span.attributes[TestExecutionContext.SpanAttributes.TEST_RUN_ID] == "tr-9"
            # Unmapped keys travel too, under the haystack namespace.
            assert span.attributes["haystack.invocation.user_id"] == "u-1"

        assert exporter.get_finished_spans()[-1].attributes[attrs.IS_TURN_ROOT] is True
        for child in children:
            assert attrs.IS_TURN_ROOT not in child.attributes

    def test_socket_context_reaches_the_root_and_later_components(self):
        """
        The input socket supplies the context from inside the run, so it cannot reach backwards.

        A component whose span closed before the connector executed is already exported. The root
        span always gets it — it closes last — which is what conversation grouping needs. Callers
        who want it on every span use `rhesis_invocation_context` instead, as the README says.
        """
        pipe, exporter = self._traced_pipeline(_Echo(), "echo")
        pipe.run(
            {
                "echo": {"messages": [ChatMessage.from_user("hi")]},
                "tracer": {"invocation_context": {"session_id": "dave"}},
            }
        )

        root = self._root_attributes(exporter)
        assert root[ConversationContext.SpanAttributes.CONVERSATION_ID] == "dave"

    def test_context_lands_without_any_chat_messages(self):
        """
        Test-run correlation must not depend on the pipeline happening to carry chat messages.

        The mapped attributes used to be stamped only alongside extracted conversation text, so a
        pipeline of plain components dropped ``invocation_context`` entirely.
        """
        pipe, exporter = self._traced_pipeline(_Upper(), "up")
        pipe.run(
            {
                "up": {"text": "hello"},
                "tracer": {"invocation_context": {"session_id": "carol", "test_run_id": "tr-1"}},
            }
        )
        attributes = self._root_attributes(exporter)

        assert attributes[ConversationContext.SpanAttributes.CONVERSATION_ID] == "carol"
        assert attributes[TestExecutionContext.SpanAttributes.TEST_RUN_ID] == "tr-1"
