# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rhesis.telemetry.constants import ConversationContext, TestExecutionContext

from haystack_integrations.components.connectors.rhesis import RhesisConnector
from haystack_integrations.tracing.rhesis import DefaultSpanHandler

_PROVIDER_PATH = "haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span, component_type=None):
        pass


class TestRhesisConnector:
    def test_run(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
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
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
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
                "base_url": "http://localhost:8080",
                "project_id": None,
                "environment": "development",
                "frontend_url": None,
                "span_handler": None,
            },
        }

    def test_to_dict_with_custom_handler(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
            connector = RhesisConnector(name="Chat example", span_handler=CustomSpanHandler())
            serialized = connector.to_dict()

        assert serialized["init_parameters"]["span_handler"]["type"].endswith("CustomSpanHandler")

    def test_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
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

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("RHESIS_API_KEY", raising=False)
        with (
            patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"),
            pytest.raises(Exception),
        ):
            RhesisConnector(name="Chat example", api_key=Secret.from_env_var("RHESIS_API_KEY"))

    def test_pipeline_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
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
            patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"),
            patch(
                "haystack_integrations.components.connectors.rhesis.rhesis_connector.tracing.enable_tracing"
            ) as mock_enable,
        ):
            RhesisConnector(name="Chat example")
            mock_enable.assert_called_once()


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
