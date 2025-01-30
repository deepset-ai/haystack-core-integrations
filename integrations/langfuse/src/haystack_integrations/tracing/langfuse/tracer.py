import contextlib
import os
from typing import Any, Dict, Iterator, Optional, Union

from haystack.components.generators.openai_utils import _convert_message_to_openai_format
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer, tracer
from haystack.tracing import utils as tracing_utils

import langfuse

HAYSTACK_LANGFUSE_ENFORCE_FLUSH_ENV_VAR = "HAYSTACK_LANGFUSE_ENFORCE_FLUSH"
_SUPPORTED_GENERATORS = [
    "AzureOpenAIGenerator",
    "OpenAIGenerator",
    "AnthropicGenerator",
    "HuggingFaceAPIGenerator",
    "HuggingFaceLocalGenerator",
    "CohereGenerator",
]
_SUPPORTED_CHAT_GENERATORS = [
    "AzureOpenAIChatGenerator",
    "OpenAIChatGenerator",
    "AnthropicChatGenerator",
    "HuggingFaceAPIChatGenerator",
    "HuggingFaceLocalChatGenerator",
    "CohereChatGenerator",
]
_ALL_SUPPORTED_GENERATORS = _SUPPORTED_GENERATORS + _SUPPORTED_CHAT_GENERATORS


class LangfuseSpan(Span):
    """
    Internal class representing a bridge between the Haystack span tracing API and Langfuse.
    """

    def __init__(self, span: "Union[langfuse.client.StatefulSpanClient, langfuse.client.StatefulTraceClient]") -> None:
        """
        Initialize a LangfuseSpan instance.

        :param span: The span instance managed by Langfuse.
        """
        self._span = span
        # locally cache tags
        self._data: Dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a generic tag for this span.

        :param key: The tag key.
        :param value: The tag value.
        """
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.update(metadata={key: coerced_value})
        self._data[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
        """
        Set a content-specific tag for this span.

        :param key: The content tag key.
        :param value: The content tag value.
        """
        if not tracer.is_content_tracing_enabled:
            return
        if key.endswith(".input"):
            if "messages" in value:
                messages = [_convert_message_to_openai_format(m) for m in value["messages"]]
                self._span.update(input=messages)
            else:
                self._span.update(input=value)
        elif key.endswith(".output"):
            if "replies" in value:
                if all(isinstance(r, ChatMessage) for r in value["replies"]):
                    replies = [_convert_message_to_openai_format(m) for m in value["replies"]]
                else:
                    replies = value["replies"]
                self._span.update(output=replies)
            else:
                self._span.update(output=value)

        self._data[key] = value

    def raw_span(self) -> Any:
        """
        Return the underlying span instance.

        :return: The Langfuse span instance.
        """
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {}


class LangfuseTracer(Tracer):
    """
    Internal class representing a bridge between the Haystack tracer and Langfuse.
    """

    def __init__(self, tracer: "langfuse.Langfuse", name: str = "Haystack", public: bool = False) -> None:
        """
        Initialize a LangfuseTracer instance.

        :param tracer: The Langfuse tracer instance.
        :param name: The name of the pipeline or component. This name will be used to identify the tracing run on the
            Langfuse dashboard.
        :param public: Whether the tracing data should be public or private. If set to `True`, the tracing data will
        be publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private
        and only accessible to the Langfuse account owner.
        """
        self._tracer = tracer
        self._context: list[LangfuseSpan] = []
        self._name = name
        self._public = public
        self.enforce_flush = os.getenv(HAYSTACK_LANGFUSE_ENFORCE_FLUSH_ENV_VAR, "true").lower() == "true"

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        """
        Start and manage a new trace span.
        :param operation_name: The name of the operation.
        :param tags: A dictionary of tags to attach to the span.
        :return: A context manager yielding the span.
        """
        tags = tags or {}
        span_name = tags.get("haystack.component.name", operation_name)

        if tags.get("haystack.component.type") in _ALL_SUPPORTED_GENERATORS:
            span = LangfuseSpan(self.current_span().raw_span().generation(name=span_name))
        else:
            span = LangfuseSpan(self.current_span().raw_span().span(name=span_name))

        self._context.append(span)
        span.set_tags(tags)

        yield span

        if tags.get("haystack.component.type") in _SUPPORTED_GENERATORS:
            meta = span._data.get("haystack.component.output", {}).get("meta")
            if meta:
                # Haystack returns one meta dict for each message, but the 'usage' value
                # is always the same, let's just pick the first item
                m = meta[0]
                span._span.update(usage=m.get("usage") or None, model=m.get("model"))
        elif tags.get("haystack.component.type") in _SUPPORTED_CHAT_GENERATORS:
            replies = span._data.get("haystack.component.output", {}).get("replies")
            if replies:
                meta = replies[0].meta
                span._span.update(usage=meta.get("usage") or None, model=meta.get("model"))

        pipeline_input = tags.get("haystack.pipeline.input_data", None)
        if pipeline_input:
            span._span.update(input=tags["haystack.pipeline.input_data"])
        pipeline_output = tags.get("haystack.pipeline.output_data", None)
        if pipeline_output:
            span._span.update(output=tags["haystack.pipeline.output_data"])

        span.raw_span().end()
        self._context.pop()

        if len(self._context) == 1:
            # The root span has to be a trace, which need to be removed from the context after the pipeline run
            self._context.pop()

            if self.enforce_flush:
                self.flush()

    def flush(self):
        self._tracer.flush()

    def current_span(self) -> Span:
        """
        Return the currently active span.

        :return: The currently active span.
        """
        if not self._context:
            # The root span has to be a trace
            self._context.append(LangfuseSpan(self._tracer.trace(name=self._name, public=self._public)))
        return self._context[-1]

    def get_trace_url(self) -> str:
        """
        Return the URL to the tracing data.
        :return: The URL to the tracing data.
        """
        return self._tracer.get_trace_url()
