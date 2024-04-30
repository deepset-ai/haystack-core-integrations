from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterator

from haystack.tracing import Span, Tracer

from langfuse_haystack.tracing.langfuse_tracing import LangfuseSpan, LangfuseTrace, TraceContextManager

if TYPE_CHECKING:
    from langfuse import Langfuse


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
