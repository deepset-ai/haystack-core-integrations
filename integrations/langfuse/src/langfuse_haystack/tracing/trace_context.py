from dataclasses import dataclass, field
import threading
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from langfuse.client import StatefulClient

@dataclass
class TraceContext:
    trace_id: str
    inputs: dict[str, Any] | None = field(default_factory=dict)
    outputs: dict[str, Any] | None = field(default_factory=dict)
    current_span: str | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)


class TraceContextManager:
    context = threading.local()

    @classmethod
    def add(cls, span_or_trace: StatefulClient):
        if not hasattr(cls.context, "langfuse_trace"):
            cls.context.langfuse_trace = []

        cls.context.langfuse_trace.append(TraceContext(span_or_trace.id))

    @classmethod
    def get(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.context.langfuse_trace:
            return cls.context.langfuse_trace[-1]
        return None

    @classmethod
    def get_trace_id(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.context.langfuse_trace:
            return cls.context.langfuse_trace[0].id
        return None

    @classmethod
    def is_active(cls):
        return hasattr(cls.context, "langfuse_trace") and cls.context.langfuse_trace

    @classmethod
    def add_input(cls, data: dict[str, Any]):
        if hasattr(cls.context, "langfuse_trace"):
            cls.context.langfuse_trace[-1].inputs.update(data)

    @classmethod
    def add_output(cls, output: dict[str, Any]):
        if hasattr(cls.context, "langfuse_trace"):
            cls.context.langfuse_trace[-1].outputs.update(output)

    @classmethod
    def add_metadata(cls, metadata: dict[str, Any]):
        if hasattr(cls.context, "langfuse_trace"):
            cls.context.langfuse_trace[-1].metadata = metadata

    @classmethod
    def get_metadata(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.has_metadata():
            return cls.context.langfuse_trace[-1].metadata
        return None

    @classmethod
    def has_metadata(cls):
        if hasattr(cls.context, "langfuse_trace"):
            return cls.context.langfuse_trace[-1].metadata is not None
        return False

    @classmethod
    def add_current_span(cls, span: Span):
        if hasattr(cls.context, "langfuse_trace"):
            cls.context.langfuse_trace[-1].current_span = span

    @classmethod
    def has_current_span(cls):
        if hasattr(cls.context, "langfuse_trace"):
            return cls.context.langfuse_trace[-1].current_span is not None
        return False

    @classmethod
    def get_current_span(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.has_current_span():
            return cls.context.langfuse_trace[-1].current_span
        return None

    @classmethod
    def remove_current_span(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.has_current_span():
            cls.context.langfuse_trace[-1].current_span = None

    @classmethod
    def has_input(cls):
        if hasattr(cls.context, "langfuse_trace"):
            return cls.context.langfuse_trace[-1].inputs is not None
        return False

    @classmethod
    def has_output(cls):
        if hasattr(cls.context, "langfuse_trace"):
            return cls.context.langfuse_trace[-1].outputs is not None
        return False

    @classmethod
    def get_input(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.has_input():
            return cls.context.langfuse_trace[-1].inputs
        return None

    @classmethod
    def get_output(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.has_output():
            return cls.context.langfuse_trace[-1].outputs
        return None

    @classmethod
    def remove(cls):
        cls.context.langfuse_trace.pop()

    @classmethod
    def parent_id(cls):
        if hasattr(cls.context, "langfuse_trace") and len(cls.context.langfuse_trace) > 1:
            return cls.context.langfuse_trace[-1].id
        return None
