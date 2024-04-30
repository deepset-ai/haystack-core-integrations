from langfuse_haystack.tracing.langfuse_tracing import TraceContextManager
import pytest
from unittest.mock import patch, Mock
from langfuse_haystack.tracing.tracer import LangfuseTracer


class TestLangfuseTracer:
    @pytest.fixture
    def mock_langfuse(self):
        return Mock()

    @pytest.fixture
    def tracer(self, mock_langfuse):
        return LangfuseTracer(mock_langfuse)

    @patch('langfuse_haystack.tracing.tracer.TraceContextManager')
    def test_trace_pipeline_active(self, mock_trace_context_manager, tracer):
        mock_trace_context_manager.is_active.return_value = True
        operation_name = "pipeline_operation"
        with tracer.trace(operation_name) as span:
            assert span is None
        
        mock_trace_context_manager.add.assert_called_once()
        mock_trace_context_manager.remove.assert_called_once()

    @patch('langfuse_haystack.tracing.tracer.LangfuseTrace')
    @patch('langfuse_haystack.tracing.tracer.TraceContextManager')
    def test_trace_pipeline_not_active(self, mock_trace_context_manager, mock_langfuse_trace, tracer):
        mock_trace_context_manager.is_active.return_value = False
        operation_name = "pipeline_operation"
        tags = {"tag1": "value1"}
        with tracer.trace(operation_name, tags):
            mock_langfuse_trace.assert_called_once_with(tracer.langfuse, name=operation_name, tags=tags)

    @patch('langfuse_haystack.tracing.tracer.LangfuseSpan')
    def test_trace_span_for_component(self, mock_langfuse_span, tracer):
        operation_name = "component"
        tags = {"tag1": "value1"}
        with tracer.trace(operation_name, tags):
            mock_langfuse_span.assert_called_once_with(tracer.langfuse, tags)

    @patch('langfuse_haystack.tracing.tracer.TraceContextManager')
    def test_current_span(self, mock_trace_context_manager, tracer):
        tracer.current_span()
        mock_trace_context_manager.get_current_span.assert_called()