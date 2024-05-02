import contextlib
from typing import Any, Dict, Iterator, Optional, Union

from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer, tracer
from haystack.tracing import utils as tracing_utils

import langfuse


class LangfuseSpan(Span):
    def __init__(self, span: "Union[langfuse.client.StatefulSpanClient, langfuse.client.StatefulTraceClient]") -> None:
        self._span = span
        # locally cache tags
        self._data: Dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.update(metadata={key: coerced_value})
        self._data[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
        if not tracer.is_content_tracing_enabled:
            return
        if key.endswith(".input"):
            if "messages" in value:
                messages = [self.to_openai_format(m) for m in value["messages"]]
                self._span.update(input=messages)
            else:
                self._span.update(input=value)
        elif key.endswith(".output"):
            if "replies" in value:
                if all(isinstance(r, ChatMessage) for r in value["replies"]):
                    replies = [self.to_openai_format(m) for m in value["replies"]]
                else:
                    replies = value["replies"]
                self._span.update(output=replies)
            else:
                self._span.update(output=value)

        self._data[key] = value

    def raw_span(self) -> Any:
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {}

    def to_openai_format(self, m: ChatMessage) -> Dict[str, Any]:
        """
        Remove after haystack 2.0.1 has been released and use the `to_openai_format` method from the ChatMessage class
        """
        msg = {"role": m.role.value, "content": m.content}
        if m.name:
            msg["name"] = m.name

        return msg


class LangfuseTracer(Tracer):
    def __init__(self, tracer: "langfuse.Langfuse", name: str = "Haystack", public: bool = False) -> None:
        self._tracer = tracer
        self._context: list[LangfuseSpan] = []
        self._name = name
        self._public = public

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        tags = tags or {}
        span_name = tags.get("haystack.component.name", operation_name)

        if tags.get("haystack.component.type") in ["OpenAIGenerator", "OpenAIChatGenerator"]:
            span = LangfuseSpan(self.current_span().raw_span().generation(name=span_name))
        else:
            span = LangfuseSpan(self.current_span().raw_span().span(name=span_name))

        self._context.append(span)
        span.set_tags(tags)

        yield span

        if tags.get("haystack.component.type") == "OpenAIGenerator":
            meta = span._data.get("haystack.component.output", {}).get("meta")
            if meta:
                # Haystack returns one meta dict for each message, but the 'usage' value
                # is always the same, let's just pick the first item
                m = meta[0]
                span._span.update(usage=m.get("usage"), model=m.get("model"))
        elif tags.get("haystack.component.type") == "OpenAIChatGenerator":
            replies = span._data.get("haystack.component.output", {}).get("replies")
            if replies:
                meta = replies[0].meta
                span._span.update(usage=meta.get("usage"), model=meta.get("model"))

        pipeline_input = tags.get("haystack.pipeline.input_data", None)
        if pipeline_input:
            span._span.update(input=tags["haystack.pipeline.input_data"])
        pipeline_output = tags.get("haystack.pipeline.output_data", None)
        if pipeline_output:
            span._span.update(output=tags["haystack.pipeline.output_data"])

        span.raw_span().end()
        self._context.pop()
        self._tracer.flush()

    def current_span(self) -> Span:
        if not self._context:
            # The root span has to be a trace
            self._context.append(LangfuseSpan(self._tracer.trace(name=self._name, public=self._public)))
        return self._context[-1]

    def get_trace_url(self) -> str:
        return self._tracer.get_trace_url()
