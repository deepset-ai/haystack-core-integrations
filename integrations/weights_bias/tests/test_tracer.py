import logging
from unittest.mock import patch

from weave.trace.autopatch import AutopatchSettings

from haystack_integrations.tracing.weave.tracer import WeaveSpan, WeaveTracer


class MockCall:
    def __init__(self, op=None, inputs=None, attributes=None, display_name=None, parent=None):
        self.id = "test_id"
        self.run_id = "test_run_id"
        self.op = op
        self.inputs = inputs or {}
        self.attributes = attributes or {}
        self.display_name = display_name
        self.parent = parent


class MockWeaveClient:
    @staticmethod
    def create_call(op=None, inputs=None, attributes=None, display_name=None, parent=None):
        return MockCall(op=op, inputs=inputs, attributes=attributes, display_name=display_name, parent=parent)

    def finish_call(self, call, output=None, exception=None):
        pass


class TestWeaveTracer:

    def test_initialization(self):
        with patch("weave.init") as mock_init:
            mock_init.return_value = MockWeaveClient()
            tracer = WeaveTracer(project_name="test_project")
            assert tracer._client is not None
            assert tracer._current_span is None
            mock_init.assert_called_once_with("test_project")

    def test_initialization_with_weave_init_kwargs(self):
        with patch("weave.init") as mock_init:
            mock_init.return_value = MockWeaveClient()
            tracer = WeaveTracer(
                project_name="test_project",
                autopatch_settings=AutopatchSettings(disable_autopatch=True),
            )
            assert tracer._client is not None
            assert tracer._current_span is None
            mock_init.assert_called_once_with(
                "test_project", autopatch_settings=AutopatchSettings(disable_autopatch=True)
            )

    def test_create_new_span(self):
        with patch("weave.init") as mock_init:
            mock_client = MockWeaveClient()
            mock_init.return_value = mock_client
            tracer = WeaveTracer(project_name="test_project")
            with tracer.trace("operation_name", tags={"tag1": "value1", "tag2": "value2"}) as span:
                assert isinstance(span, WeaveSpan)
                assert span.raw_span() is not None
                assert tracer.current_span() == span

            assert tracer.current_span() is None

    def test_component_run_span(self):
        with patch("weave.init") as mock_init:
            mock_client = MockWeaveClient()
            mock_init.return_value = mock_client
            tracer = WeaveTracer(project_name="test_project")
            component_tags = {
                "haystack.component.name": "TestComponent",
                "haystack.component.type": "test_type",
                "haystack.component.input": {"input_key": "input_value"},
                "haystack.component.output": {"output_key": "output_value"},
            }

            with tracer.trace("haystack.component.run", tags=component_tags) as span:
                assert isinstance(span, WeaveSpan)
                assert tracer.current_span() == span

            assert tracer.current_span() is None

    def test_exception_handling(self):
        with patch("weave.init") as mock_init:
            mock_client = MockWeaveClient()
            mock_init.return_value = mock_client
            tracer = WeaveTracer(project_name="test_project")
            try:
                with tracer.trace("operation_name") as span:  # noqa
                    msg = "Test error"
                    raise ValueError(msg)
            except ValueError:
                assert tracer.current_span() is None

    def test_init_with_tracing_disabled(self, monkeypatch, caplog):
        # unset environment variable HAYSTACK_CONTENT_TRACING_ENABLED
        monkeypatch.delenv("HAYSTACK_CONTENT_TRACING_ENABLED", raising=False)
        with caplog.at_level(logging.WARNING, logger="haystack_integrations.tracing.weave.tracer"):
            with patch("weave.init") as mock_init:
                mock_init.return_value = MockWeaveClient()
                WeaveTracer(project_name="test_project")
                assert (
                    "Inputs and Outputs of components traces will not be logged because Haystack tracing is "
                    "disabled.To enable, set the HAYSTACK_CONTENT_TRACING_ENABLED environment variable to true "
                    "before importing Haystack.\n"
                ) in caplog.text
