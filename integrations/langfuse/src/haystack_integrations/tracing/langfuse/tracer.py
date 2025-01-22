import contextlib
import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from haystack import default_from_dict, default_to_dict, logging
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

    def __init__(self, span: "langfuse.client.StatefulClient") -> None:
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

    def raw_span(self) -> "langfuse.client.StatefulClient":
        """
        Return the underlying span instance.

        :return: The Langfuse span instance.
        """
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {}


class SpanHandler(ABC):
    """
    Abstract base class for handling Langfuse spans.
    Extend this class to customize how spans are created and processed.

    This class defines the interface for span creation and processing in Langfuse tracing.
    Custom implementations can override the span creation logic and how metadata is extracted
    and attached to spans and thus to Langfuse traces.
    """

    def __init__(self):
        self.tracer: Optional[langfuse.Langfuse] = None

    def init_tracer(self, tracer: "langfuse.Langfuse") -> None:
        """Initialize with Langfuse tracer. Called by LangfuseTracer."""
        self.tracer = tracer

    @abstractmethod
    def create_span(
        self,
        name: str,
        operation_name: str,
        parent_span: Optional[Span],
        component_type: Optional[str],
        tags: Optional[Dict[str, Any]],
    ) -> LangfuseSpan:
        """
        Create a span of appropriate type based on component.

        :param name: The name of the span
        :param operation_name: The name of the operation that created the span
        :param parent_span: The parent span if any
        :param component_type: The type of the component creating the span
        :param tags: Additional tags for the span
        :returns: A new LangfuseSpan instance
        """
        pass

    @abstractmethod
    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        """
        Process a span after it has been yielded by attaching metadata and statistics.

        Can be used to attach various types of metadata to spans, such as:
        - token usage statistics
        - model information
        - timing data (e.g., time-to-first-token in LLMs)
        - custom metrics and observations

        :param span: The LangfuseSpan that was yielded
        :param component_type: The type of the component that created this span. Used to determine
            the metadata extraction logic
        """
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanHandler":
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)


class DefaultSpanHandler(SpanHandler):
    """DefaultSpanHandler provides the default Langfuse tracing behavior for Haystack."""

    def create_span(
        self,
        name: str,
        operation_name: str,
        parent_span: Optional[Span],
        component_type: Optional[str],
        tags: Optional[Dict[str, Any]] = None,
    ) -> LangfuseSpan:
        message = "Tracer is not initialized"
        if self.tracer is None:
            raise RuntimeError(message)
        context = tracing_context_var.get({})
        tags = tags or {}
        if not parent_span:
            if operation_name != _PIPELINE_RUN_KEY:
                logger.warning(
                    "Creating a new trace without a parent span is not recommended for operation '{operation_name}'.",
                    operation_name=operation_name,
                )
            # Create a new trace when there's no parent span
            return LangfuseSpan(
                self.tracer.trace(
                    name=name,
                    id=context.get("trace_id"),
                    user_id=context.get("user_id"),
                    session_id=context.get("session_id"),
                    tags=context.get("tags"),
                    version=context.get("version"),
                )
            )
        elif component_type in _ALL_SUPPORTED_GENERATORS:
            return LangfuseSpan(parent_span.raw_span().generation(name=name))
        else:
            return LangfuseSpan(parent_span.raw_span().span(name=name))

    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        if component_type in _SUPPORTED_GENERATORS:
            meta = span._data.get(_COMPONENT_OUTPUT_KEY, {}).get("meta")
            if meta:
                m = meta[0]
                span._span.update(usage=m.get("usage") or None, model=m.get("model"))
        elif component_type in _SUPPORTED_CHAT_GENERATORS:
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


class LangfuseTracer(Tracer):
    """
    Internal class representing a bridge between the Haystack tracer and Langfuse.
    """

    def __init__(
        self,
        tracer: "langfuse.Langfuse",
        name: str = "Haystack",
        public: bool = False,
        span_handler: Optional[SpanHandler] = None,
    ) -> None:
        """
        Initialize a LangfuseTracer instance.

        :param tracer: The Langfuse tracer instance.
        :param name: The name of the pipeline or component. This name will be used to identify the tracing run on the
            Langfuse dashboard.
        :param public: Whether the tracing data should be public or private. If set to `True`, the tracing data will
            be publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private
            and only accessible to the Langfuse account owner.
        :param span_handler: Custom handler for processing spans. If None, uses DefaultSpanHandler.
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
        self._span_handler = span_handler or DefaultSpanHandler()
        self._span_handler.init_tracer(tracer)

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        tags = tags or {}
        span_name = tags.get(_COMPONENT_NAME_KEY, operation_name)
        component_type = tags.get(_COMPONENT_TYPE_KEY)

        # Create span using the handler
        span = self._span_handler.create_span(
            name=span_name,
            operation_name=operation_name,
            parent_span=parent_span,
            component_type=component_type,
            tags=tags,
        )

        self._context.append(span)
        span.set_tags(tags)

        yield span

        # Let the span handler process the span
        self._span_handler.handle(span, component_type)

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
