from unittest.mock import Mock, MagicMock, patch

from haystack_integrations.tracing.langfuse.tracer import LangfuseTracer


class TestLangfuseTracer:

    #  LangfuseTracer can be initialized with a Langfuse instance, a name and a boolean value for public.
    def test_initialization(self):
        langfuse_instance = Mock()
        tracer = LangfuseTracer(tracer=langfuse_instance, name="Haystack", public=True)
        assert tracer._tracer == langfuse_instance
        assert tracer._context == []
        assert tracer._name == "Haystack"
        assert tracer._public

    # check that the trace method is called on the tracer instance with the provided operation name and tags
    # check that the span is added to the context and removed after the context manager exits
    def test_create_new_span(self):
        mock_raw_span = MagicMock()
        mock_raw_span.operation_name = "operation_name"
        mock_raw_span.metadata = {"tag1": "value1", "tag2": "value2"}

        with patch("haystack_integrations.tracing.langfuse.tracer.LangfuseSpan") as MockLangfuseSpan:
            mock_span_instance = MockLangfuseSpan.return_value
            mock_span_instance.raw_span.return_value = mock_raw_span

            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value = mock_span_instance

            mock_tracer = MagicMock()
            mock_tracer.trace.return_value = mock_context_manager

            tracer = LangfuseTracer(tracer=mock_tracer, name="Haystack", public=False)

            with tracer.trace("operation_name", tags={"tag1": "value1", "tag2": "value2"}) as span:
                assert len(tracer._context) == 2, "The trace span should have been added to the the root context span"
                assert span.raw_span().operation_name == "operation_name"
                assert span.raw_span().metadata == {"tag1": "value1", "tag2": "value2"}

            assert len(tracer._context) == 1, "The trace span should have been popped, leaving root span in the context"

    # check that update method is called on the span instance with the provided key value pairs
    def test_update_span_with_pipeline_input_output_data(self):
        class MockTracer:

            def trace(self, name, **kwargs):
                return MockSpan()

            def flush(self):
                pass

        class MockSpan:
            def __init__(self):
                self._data = {}
                self._span = self
                self.operation_name = "operation_name"

            def raw_span(self):
                return self

            def span(self, name=None):
                # assert correct operation name passed to the span
                assert name == "operation_name"
                return self

            def update(self, **kwargs):
                self._data.update(kwargs)

            def generation(self, name=None):
                return self

            def end(self):
                pass

        tracer = LangfuseTracer(tracer=MockTracer(), name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}) as span:
            assert span.raw_span()._data["metadata"] == {"haystack.pipeline.input_data": "hello"}

        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.output_data": "bye"}) as span:
            assert span.raw_span()._data["metadata"] == {"haystack.pipeline.output_data": "bye"}
