from __future__ import annotations

import contextlib
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from haystack.tracing import Span, Tracer
from haystack.tracing import utils as tracing_utils
from langfuse_haystack.tracing.context import TraceContextManager

if TYPE_CHECKING:
    from langfuse import Langfuse


session_context = threading.local()
session_context.id = None


@contextlib.contextmanager
def langfuse_session():
    session_context.id = uuid.uuid4().hex
    yield
    session_context.id = None


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
    def __init__(self, langfuse: Langfuse, tags: dict[str, Any] | None = None) -> None:
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
                # other items of the value dict will be stored in meta
                meta = {k: v for k, v in value.items() if k != "prompt"}
                self._span.update(input=value["prompt"], meta=meta)
            else:
                self._span.update(input=value)

            TraceContextManager.add_input({self.component_name: value})

        elif key == "haystack.component.output":
            if "replies" in value:
                meta = {k: v for k, v in value.items() if k != "replies"}
                self._span.update(output=value["replies"][0], meta=meta)
            else:
                self._span.update(output=value)
            TraceContextManager.add_output({self.component_name: value})

            if "meta" in value:
                meta = value["meta"][-1]
                if "model" in meta:
                    self._span.update(model=meta["model"])
                if "usage" in meta:
                    self._span.update(usage=meta["usage"])
                TraceContextManager.add_metadata(meta)

        self._span.update(tags={key: coerced_value})

    def _is_generation_span(self, component_name: str) -> bool:
        return component_name.endswith(("Embedder", "Generator"))


class LangfuseTracer(Tracer):
    def __init__(self, langfuse: Langfuse) -> None:
        self.langfuse = langfuse

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: dict[str, Any] | None = None) -> Iterator[Span]:
        if "pipeline" in operation_name:
            if TraceContextManager.is_active():
                TraceContextManager.add(self.current_span())
                yield
                TraceContextManager.remove()
            else:
                with LangfuseTrace(self.langfuse, name=operation_name, tags=tags):
                    yield

        else:
            with LangfuseSpan(self.langfuse, tags) as span:
                yield span

    def current_span(self) -> Span | None:
        return TraceContextManager.get_current_span()
