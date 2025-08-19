import os
import pytest

# Enable content tracing for tests - must be set before importing haystack modules
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack_integrations.tracing.langfuse.tracer import (
    tracing_context_var,
    span_stack_var,
    root_trace_client_var,
)


@pytest.fixture(autouse=True)
def reset_tracing_globals():
    """Ensure tracing-related ContextVars are clean for every test.

    This prevents cross-test contamination where roots/spans leak via ContextVars
    or per-task stacks, causing order-dependent flakiness.
    """
    token_ctx = tracing_context_var.set({})
    token_stack = span_stack_var.set(None)
    token_root = root_trace_client_var.set(None)
    try:
        yield
    finally:
        tracing_context_var.reset(token_ctx)
        span_stack_var.reset(token_stack)
        root_trace_client_var.reset(token_root)


