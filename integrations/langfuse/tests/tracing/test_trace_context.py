import pytest
from unittest import mock
from langfuse.client import StatefulClient
from langfuse_haystack.tracing.langfuse_tracing import LangfuseTrace, TraceContextManager, LangfuseSpan, langfuse_session

from haystack.tracing import Span


@pytest.fixture(autouse=True)
def reset_context():
    TraceContextManager.reset()

@pytest.fixture
def mock_trace():
    client = mock.Mock(spec=StatefulClient)
    client.id = "trace_id"
    return client

@pytest.fixture
def mock_stateful_client():
    client = mock.Mock(spec=StatefulClient)
    client.id = "stateful_client_id"
    return client

@pytest.fixture
def mock_span():
    span = mock.Mock(spec=Span)
    return span

@pytest.fixture
def mock_langfuse():
    return mock.Mock()

@pytest.fixture
def mock_tags():
    return {"haystack.component.name": "TestComponent", "haystack.component.type": "TestType"}

@pytest.fixture
def mock_generator_tags():
    return {"haystack.component.name": "llm", "haystack.component.type": "TestComponentGenerator"}

@pytest.fixture
def mock_embedder_tags():
    return {"haystack.component.name": "embeddings", "haystack.component.type": "TestComponentEmbedder"}

class TestTraceContextManager:
    @pytest.fixture(autouse=True)
    def clear_context(self):
        while TraceContextManager.is_active():
            TraceContextManager.remove()

    def test_add(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        assert TraceContextManager.context.langfuse_trace[-1].trace_id == "stateful_client_id"

    def test_get(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        assert TraceContextManager.get().trace_id == "stateful_client_id"

    def test_get_trace_id(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        assert TraceContextManager.get_trace_id() == "stateful_client_id"

    def test_is_active(self, mock_stateful_client):
        assert not TraceContextManager.is_active()
        TraceContextManager.add(mock_stateful_client)
        assert TraceContextManager.is_active()

    def test_is_active(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_input({"test": "input"})
        assert TraceContextManager.context.langfuse_trace[-1].inputs["test"] == "input"

    def test_add_output(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_output({"test": "output"})
        assert TraceContextManager.context.langfuse_trace[-1].outputs["test"] == "output"

    def test_add_metadata(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_metadata({"test": "metadata"})
        assert TraceContextManager.context.langfuse_trace[-1].metadata["test"] == "metadata"

    def test_get_metadata(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_metadata({"test": "metadata"})
        assert TraceContextManager.get_metadata()["test"] == "metadata"

    def test_has_metadata(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        assert not TraceContextManager.has_metadata()
        TraceContextManager.add_metadata({"test": "metadata"})
        assert TraceContextManager.has_metadata()

    def test_add_current_span(self, mock_span, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_current_span(mock_span)
        assert TraceContextManager.context.langfuse_trace[-1].current_span == mock_span

    def test_has_current_span(self, mock_span, mock_stateful_client):
        assert not TraceContextManager.has_current_span()
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_current_span(mock_span)
        assert TraceContextManager.has_current_span()

    def test_get_current_span(self, mock_span, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_current_span(mock_span)
        assert TraceContextManager.get_current_span() == mock_span

    def test_remove_current_span(self, mock_span, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_current_span(mock_span)
        TraceContextManager.remove_current_span()
        assert TraceContextManager.context.langfuse_trace[-1].current_span is None

    def test_has_input(self, mock_stateful_client):
        assert not TraceContextManager.has_input()
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_input({"test": "input"})
        assert TraceContextManager.has_input()

    def test_has_output(self, mock_stateful_client):
        assert not TraceContextManager.has_output()
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_output({"test": "output"})
        assert TraceContextManager.has_output()

    def test_get_input(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_input({"test": "input"})
        assert TraceContextManager.get_input()["test"] == "input"

    def test_get_output(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add_output({"test": "output"})
        assert TraceContextManager.get_output()["test"] == "output"

    def test_remove(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        assert TraceContextManager.is_active()
        TraceContextManager.remove()
        assert not TraceContextManager.is_active()

    def test_parent_id(self, mock_stateful_client):
        TraceContextManager.add(mock_stateful_client)
        TraceContextManager.add(mock_stateful_client)
        assert TraceContextManager.parent_id() == "stateful_client_id"

class TestLangfuseSpan:
    def test_init(self, mock_langfuse, mock_tags):
        span = LangfuseSpan(mock_langfuse, mock_tags)
        assert span.langfuse == mock_langfuse
        assert span.tags == mock_tags
        assert span.component_name == "TestComponent"
        assert span.component_type == "TestType"
        assert span.name == "TestComponent::TestType"

    def test_set_tag_input(self, mock_langfuse, mock_tags):
        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            span.set_tag("haystack.component.input", "test")
            mock_langfuse.span.return_value.update.assert_called_with(tags={"haystack.component.input": "test"})

    def test_set_tag_output(self, mock_langfuse, mock_tags):
        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            span.set_tag("haystack.component.output", "test")
            mock_langfuse.span.return_value.update.assert_called_with(tags={"haystack.component.output": "test"})

    def test_is_generation_span(self, mock_langfuse, mock_tags):
        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            assert span._is_generation_span("TestEmbedder") is True
            assert span._is_generation_span("TestGenerator") is True
            assert span._is_generation_span("TestOther") is False

    def test_parent_span(self, mock_langfuse, mock_tags, mock_trace, mock_stateful_client):
        TraceContextManager.add(mock_trace)
        TraceContextManager.add(mock_stateful_client)

        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            pass
        mock_langfuse.span.assert_called_with(name="TestComponent::TestType", trace_id=mock_trace.id, parent_observation_id=mock_stateful_client.id, session_id=None)

    def test_span_with_session(self, mock_langfuse, mock_tags, mock_trace, mock_stateful_client):
        TraceContextManager.add(mock_trace)
        TraceContextManager.add(mock_stateful_client)

        with langfuse_session():
            with LangfuseSpan(mock_langfuse, mock_tags) as span:
                pass
        assert span.langfuse == mock_langfuse
        assert span.langfuse.span.call_args[1]['session_id'] is not None
       
    def test_span_without_session(self, mock_langfuse, mock_tags, mock_trace):
        TraceContextManager.add(mock_trace)

        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            pass
        assert span.langfuse == mock_langfuse
        assert span.langfuse.span.call_args[1]['session_id'] is None
    
    def test_span_for_embedder_is_generation(self, mock_langfuse, mock_embedder_tags, mock_trace):
        TraceContextManager.add(mock_trace)

        with LangfuseSpan(mock_langfuse, mock_embedder_tags) as span:
            pass

        assert mock_langfuse.generation.called
        assert not mock_langfuse.span.called

    def test_span_for_generator_is_generation(self, mock_langfuse, mock_generator_tags, mock_trace):
        TraceContextManager.add(mock_trace)

        with LangfuseSpan(mock_langfuse, mock_generator_tags) as span:
            pass

        assert mock_langfuse.generation.called
        assert not mock_langfuse.span.called

    def test_span_exited_with_exception(self, mock_langfuse, mock_tags, mock_trace, mock_stateful_client):
        TraceContextManager.add(mock_trace)
        TraceContextManager.add(mock_stateful_client)

        try: 
            with LangfuseSpan(mock_langfuse, mock_tags) as span:
                raise Exception("TestException")
        except:
            pass
        assert mock_langfuse.span.return_value.update.called
        call_args = mock_langfuse.span.return_value.update.call_args[1]

        assert "ERROR" in call_args["level"]
        assert "Exception" in call_args["status_message"]

    def test_set_tag_with_prompt_value_is_extracted(self, mock_langfuse, mock_tags):
        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            span.set_tag("haystack.component.prompt", "test")
            mock_langfuse.span.return_value.update.assert_called_with(tags={"haystack.component.prompt": "test"})
    
    def test_set_tag_with_prompt_and_meta_is_extracted(self, mock_langfuse, mock_tags):
        with LangfuseSpan(mock_langfuse, mock_tags) as span:
            span.set_tag("haystack.component.prompt", {"prompt": "test", "meta": {"test": "meta"}})
            mock_langfuse.span.return_value.update.assert_called_with(tags={'haystack.component.prompt': '{"prompt": "test", "meta": {"test": "meta"}}'})
    

class TestLangfuseTrace:
    def test_init(self, mock_langfuse):
        trace = LangfuseTrace(mock_langfuse, name="TestTrace", tags={"test": "tag"})
        assert trace.langfuse == mock_langfuse
        assert trace.name == "TestTrace"
        assert trace.tags == {"test": "tag"}

    def test_trace_is_created(self, mock_langfuse):
        with LangfuseTrace(mock_langfuse, name="TestTrace"):
            pass

        assert mock_langfuse.trace.called
        assert mock_langfuse.trace.call_args[1] == {"name": "TestTrace", "session_id": None}
    
    def test_trace_is_create_with_session(self, mock_langfuse):
        with langfuse_session() as session_id:
            with LangfuseTrace(mock_langfuse, name="TestTrace"):
                pass
        assert mock_langfuse.trace.called
        assert mock_langfuse.trace.call_args[1] == {"name": "TestTrace", "session_id": session_id}
