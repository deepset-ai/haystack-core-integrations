import contextlib
import os
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union

from haystack import logging
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer
from haystack.tracing import tracer as proxy_tracer
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
_ALL_SUPPORTED_GENERATORS = _SUPPORTED_GENERATORS + _SUPPORTED_CHAT_GENERATORS

# These are the keys used by Haystack for traces and span.
# We keep them here to avoid making typos when using them.
_PIPELINE_RUN_KEY = "haystack.pipeline.run"
_COMPONENT_NAME_KEY = "haystack.component.name"
_COMPONENT_TYPE_KEY = "haystack.component.type"
_COMPONENT_OUTPUT_KEY = "haystack.component.output"

# Context var used to keep track of tracing related info.
# This mainly useful for parents spans.
tracing_context_var: ContextVar[Dict[Any, Any]] = ContextVar("tracing_context")


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
        if not proxy_tracer.is_content_tracing_enabled:
            return
        if key.endswith(".input"):
            if "messages" in value:
                messages = [m.to_openai_dict_format() for m in value["messages"]]
                self._span.update(input=messages)
            else:
                self._span.update(input=value)
        elif key.endswith(".output"):
            if "replies" in value:
                if all(isinstance(r, ChatMessage) for r in value["replies"]):
                    replies = [m.to_openai_dict_format() for m in value["replies"]]
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
        if not proxy_tracer.is_content_tracing_enabled:
            logger.warning(
                "Traces will not be logged to Langfuse because Haystack tracing is disabled. "
                "To enable, set the HAYSTACK_CONTENT_TRACING_ENABLED environment variable to true "
                "before importing Haystack."
            )
        self._tracer = tracer
        self._context: List[LangfuseSpan] = []
        self._name = name
        self._public = public
        self.enforce_flush = os.getenv(HAYSTACK_LANGFUSE_ENFORCE_FLUSH_ENV_VAR, "true").lower() == "true"

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        tags = tags or {}
        span_name = tags.get(_COMPONENT_NAME_KEY, operation_name)

        # Create new span depending whether there's a parent span or not
        if not parent_span:
            if operation_name != _PIPELINE_RUN_KEY:
                logger.warning(
                    "Creating a new trace without a parent span is not recommended for operation '{operation_name}'.",
                    operation_name=operation_name,
                )
            # Create a new trace if no parent span is provided
            context = tracing_context_var.get({})
            span = LangfuseSpan(
                self._tracer.trace(
                    name=self._name,
                    public=self._public,
                    id=context.get("trace_id"),
                    user_id=context.get("user_id"),
                    session_id=context.get("session_id"),
                    tags=context.get("tags"),
                    version=context.get("version"),
                )
            )
        elif tags.get(_COMPONENT_TYPE_KEY) in _ALL_SUPPORTED_GENERATORS:
            span = LangfuseSpan(parent_span.raw_span().generation(name=span_name))
        else:
            span = LangfuseSpan(parent_span.raw_span().span(name=span_name))

        self._context.append(span)
        span.set_tags(tags)

        yield span

        # Update span metadata based on component type
        if tags.get(_COMPONENT_TYPE_KEY) in _SUPPORTED_GENERATORS:
            # Haystack returns one meta dict for each message, but the 'usage' value
            # is always the same, let's just pick the first item
            meta = span._data.get(_COMPONENT_OUTPUT_KEY, {}).get("meta")
            if meta:
                m = meta[0]
                span._span.update(usage=m.get("usage") or None, model=m.get("model"))
        elif tags.get(_COMPONENT_TYPE_KEY) in _SUPPORTED_CHAT_GENERATORS:
            replies = span._data.get(_COMPONENT_OUTPUT_KEY, {}).get("replies")
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

        raw_span = span.raw_span()

        # In this section, we finalize both regular spans and generation spans created using the LangfuseSpan class.
        # It's important to end() these spans to ensure they are properly closed and all relevant data is recorded.
        # Note that we do not call end() on the main trace span itself, as its lifecycle is managed differently.
        if isinstance(raw_span, (langfuse.client.StatefulSpanClient, langfuse.client.StatefulGenerationClient)):
            raw_span.end()
        self._context.pop()

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
