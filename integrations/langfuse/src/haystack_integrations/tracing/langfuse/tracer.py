import contextlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union

from haystack.components.generators.openai_utils import _convert_message_to_openai_format
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer, tracer
from haystack.tracing import utils as tracing_utils

import langfuse

logger = logging.getLogger(__name__)

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
COMPONENT_TYPE_KEY = "haystack.component.type"
COMPONENT_OUTPUT_KEY = "haystack.component.output"
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

    def raw_span(self) -> "Union[langfuse.client.StatefulSpanClient, langfuse.client.StatefulTraceClient]":
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
        self._context: List[LangfuseSpan] = []
        self._name = name
        self._public = public
        self.enforce_flush = os.getenv(HAYSTACK_LANGFUSE_ENFORCE_FLUSH_ENV_VAR, "true").lower() == "true"

        root_span_raw = self._tracer.trace(name=self._name)
        self._root_span = LangfuseSpan(root_span_raw)
        # Do not add root span to self._context

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        tags = tags or {}
        span_name = tags.get("haystack.component.name", operation_name)

        # Determine if we need to create and push the root span onto the context stack
        created_root_span = False
        if not self._context:
            self._context.append(self._root_span)
            created_root_span = True

        parent_raw_span: Optional[Union[langfuse.client.StatefulSpanClient, langfuse.client.StatefulTraceClient]] = None

        if parent_span is not None:
            parent_raw_span = parent_span.raw_span()
        else:
            current_span = self.current_span()
            if current_span is not None:
                parent_raw_span = current_span.raw_span()
            else:
                parent_raw_span = None

        if tags.get(COMPONENT_TYPE_KEY) in _ALL_SUPPORTED_GENERATORS:
            new_span_raw = (
                parent_raw_span.generation(name=span_name)
                if parent_raw_span
                else self._tracer.generation(name=span_name)
            )
        elif parent_raw_span:
            new_span_raw = parent_raw_span.span(name=span_name)
        else:
            new_span_raw = self._tracer.span(name=span_name)

        span = LangfuseSpan(new_span_raw)
        self._context.append(span)
        span.set_tags(tags)

        try:
            yield span
        finally:
            self._context.pop()
            if created_root_span:
                self._context.pop()

            # Update span metadata based on component type
            if tags.get(COMPONENT_TYPE_KEY) in _SUPPORTED_GENERATORS:
                meta = span._data.get(COMPONENT_TYPE_KEY, {}).get("meta")
                if meta:
                    # Haystack returns one meta dict for each message, but the 'usage' value
                    # is always the same, let's just pick the first item
                    m = meta[0]
                    span._span.update(usage=m.get("usage") or None, model=m.get("model"))
                    # add prompt object to generator #1154
                    # if m.get("prompt_name") is not None:
                    #     prompt_name = m["prompt_name"]
                    #     prompt_obj = self.get_pipeline_run_context().get(prompt_name)
                    #     if prompt_obj:
                    #         span._span.update(prompt=prompt_obj)
                    span._span.update(usage=m.get("usage") or None, model=m.get("model"))
            elif tags.get(COMPONENT_TYPE_KEY) in _SUPPORTED_CHAT_GENERATORS:
                replies = span._data.get(COMPONENT_OUTPUT_KEY, {}).get("replies")
                if replies:
                    meta = replies[0].meta
                    completion_start_time = meta.get("completion_start_time")
                    if completion_start_time:
                        try:
                            completion_start_time = datetime.fromisoformat(completion_start_time)
                        except ValueError:
                            logger.error(f"Failed to parse completion_start_time: {completion_start_time}")
                            completion_start_time = None
                    span._span.update(
                        usage=meta.get("usage") or None,
                        model=meta.get("model"),
                        completion_start_time=completion_start_time,
                    )
            if self.enforce_flush:
                self.flush()

    def flush(self):
        self._tracer.flush()

    def current_span(self) -> Optional[Span]:
        """
        Return the current active span.

        :return: The current span if available, else None.
        """
        return self._context[-1] if self._context else None

    def get_trace_url(self) -> str:
        """
        Return the URL to the tracing data.
        :return: The URL to the tracing data.
        """
        return self._tracer.get_trace_url()
