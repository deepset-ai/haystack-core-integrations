from __future__ import annotations

import contextlib
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from haystack.tracing import Span
from haystack.tracing import utils as tracing_utils

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import StatefulClient


session_context = threading.local()
session_context.id = None


@contextlib.contextmanager
def langfuse_session():
    session_context.id = uuid.uuid4().hex
    yield session_context.id
    session_context.id = None


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
            return cls.context.langfuse_trace[0].trace_id
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
            return bool(cls.context.langfuse_trace[-1].metadata)
        return False

    @classmethod
    def add_current_span(cls, span: Span):
        if hasattr(cls.context, "langfuse_trace"):
            cls.context.langfuse_trace[-1].current_span = span

    @classmethod
    def has_current_span(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.context.langfuse_trace:
            return bool(cls.context.langfuse_trace[-1].current_span)
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
        if hasattr(cls.context, "langfuse_trace") and cls.context.langfuse_trace:
            return bool(cls.context.langfuse_trace[-1].inputs)
        return False

    @classmethod
    def has_output(cls):
        if hasattr(cls.context, "langfuse_trace") and cls.context.langfuse_trace:
            return bool(cls.context.langfuse_trace[-1].outputs)
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
            return cls.context.langfuse_trace[-1].trace_id
        return None

    @classmethod
    def reset(cls):
        cls.context = threading.local()


class LangfuseTrace:
    def __init__(self, langfuse: Langfuse, name: str, tags: dict[str, Any] | None = None) -> None:
        self.langfuse = langfuse
        self.name = name
        self.tags = tags

    def __enter__(self):
        self.trace = self.langfuse.trace(name=self.name, session_id=session_context.id)
        TraceContextManager.add(self.trace)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_tb and exc_val:
            self.trace.update(tags=["Failed", repr(exc_type)])
        if TraceContextManager.has_input():
            self.trace.update(input=TraceContextManager.get_input())

        if TraceContextManager.has_output():
            self.trace.update(output=TraceContextManager.get_output())

        if TraceContextManager.has_metadata():
            self.trace.update(metadata=TraceContextManager.get_metadata())
        TraceContextManager.remove()


class LangfuseSpan(Span):
    def __init__(self, langfuse: Langfuse, tags: dict[str, Any]) -> None:
        self.langfuse = langfuse
        self.tags = tags
        self.component_name = self.tags.pop("haystack.component.name", None)
        self.component_type = self.tags.pop("haystack.component.type", None)
        self.name = self.component_name + "::" + self.component_type

    def __enter__(self):
        trace_id = TraceContextManager.get_trace_id()

        span_params = {
            "name": self.name,
            "trace_id": trace_id,
            "parent_observation_id": TraceContextManager.parent_id(),
            "session_id": session_context.id,
        }

        if self._is_generation_span(self.component_type):
            self._span = self.langfuse.generation(**span_params)
        else:
            self._span = self.langfuse.span(**span_params)

        if self.tags:
            self.set_tags(self.tags)

        TraceContextManager.add_current_span(self._span)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_tb and exc_val:
            stack_trace = traceback.format_exception(exc_type, exc_val, exc_tb)
            status_message = repr(exc_val) + "\n" + "".join(stack_trace)
            self._span.update(level="ERROR", status_message=status_message)
        self._span.end()
        TraceContextManager.remove_current_span()

    def set_tag(self, key: str, value: Any) -> None:
        coerced_value = tracing_utils.coerce_tag_value(value)

        if key == "haystack.component.input":
            if "prompt" in value:
                meta = {k: v for k, v in value.items() if k != "prompt"}
                self._span.update(input=value["prompt"], meta=meta)
            else:
                self._span.update(input=value)

            TraceContextManager.add_input({self.component_name: value})

        elif key == "haystack.component.output":
            if "meta" in value:
                meta = {}
                if isinstance(value["meta"], list):
                    meta = value["meta"][-1]
                else:
                    meta = value["meta"]

                if "model" in meta:
                    self._span.update(model=meta["model"])
                if "usage" in meta:
                    self._span.update(usage=meta["usage"])
                TraceContextManager.add_metadata(meta)

            if "replies" in value:
                meta = {k: v for k, v in value.items() if k != "replies"}
                self._span.update(output=value["replies"][0], meta=meta)
            else:
                self._span.update(output=value)
            TraceContextManager.add_output({self.component_name: value})

        self._span.update(tags={key: coerced_value})

    def _is_generation_span(self, component_type: str) -> bool:
        return component_type.lower().endswith(("embedder", "generator"))
