import pytest
import uuid
from langfuse_haystack.tracing.langfuse_tracing import langfuse_session, TraceContextManager, LangfuseTrace, Langfuse, session_context



class MockLangfuse:
    def trace(self, name, session_id):
        return name

@pytest.fixture
def langfuse(mocker):
    mocker.patch("langfuse_haystack.tracing.langfuse_tracing.Langfuse", return_value=MockLangfuse())
    return MockLangfuse()


def test_langfuse_session():
    # Ensure there is no session id before the context manager is used
    assert session_context.id is None

    # Use the context manager and check that a session id is set
    with langfuse_session():
        assert session_context.id is not None

    # Check that the session id is unset after the context manager is used
    assert session_context.id is None

def test_trace_context_manager():
    # Test add method
    TraceContextManager.add('test')
    assert TraceContextManager.get() == 'test'

    # Test remove method
    TraceContextManager.remove()
    assert TraceContextManager.get() is None


