# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Literal, Optional, cast

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer
from haystack.tracing import tracer as proxy_tracer
from haystack.tracing import utils as tracing_utils

import langfuse
from langfuse import LangfuseSpan as LangfuseClientSpan
from langfuse.types import TraceMetadata

logger = logging.getLogger(__name__)

HAYSTACK_LANGFUSE_ENFORCE_FLUSH_ENV_VAR = "HAYSTACK_LANGFUSE_ENFORCE_FLUSH"

# These are the keys used by Haystack for traces and span.
# We keep them here to avoid making typos when using them.
_PIPELINE_INPUT_KEY = "haystack.pipeline.input_data"
_PIPELINE_OUTPUT_KEY = "haystack.pipeline.output_data"
_ASYNC_PIPELINE_RUN_KEY = "haystack.async_pipeline.run"
_PIPELINE_RUN_KEY = "haystack.pipeline.run"
_COMPONENT_NAME_KEY = "haystack.component.name"
_COMPONENT_TYPE_KEY = "haystack.component.type"
_COMPONENT_OUTPUT_KEY = "haystack.component.output"
_COMPONENT_INPUT_KEY = "haystack.component.input"

# Type alias for observation span types
ObservationSpanType = Literal["tool", "agent", "retriever", "embedding", "generation"]

# External session metadata for trace correlation (Haystack system)
# Stores trace_id, user_id, session_id, tags, version for root trace creation
tracing_context_var: ContextVar[Dict[Any, Any]] = ContextVar("tracing_context")

# Internal span execution hierarchy for our tracer
# Manages parent-child relationships and prevents cross-request span interleaving
span_stack_var: ContextVar[Optional[List["LangfuseSpan"]]] = ContextVar("span_stack", default=None)


class LangfuseSpan(Span):
    """
    Internal class representing a bridge between the Haystack span tracing API and Langfuse.
    """

    def __init__(self, context_manager: AbstractContextManager) -> None:
        """
        Initialize a LangfuseSpan instance.

        :param context_manager: The context manager from Langfuse created with
        `langfuse.get_client().start_as_current_span` or
        `langfuse.get_client().start_as_current_observation`.
        """
        self._span = context_manager.__enter__()
        self._data: Dict[str, Any] = {}
        self._context_manager = context_manager

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
                messages = [m.to_openai_dict_format(require_tool_call_ids=False) for m in value["messages"]]
                self._span.update(input=messages)
            else:
                coerced_value = tracing_utils.coerce_tag_value(value)
                self._span.update(input=coerced_value)
        elif key.endswith(".output"):
            if "replies" in value:
                if all(isinstance(r, ChatMessage) for r in value["replies"]):
                    replies = [m.to_openai_dict_format(require_tool_call_ids=False) for m in value["replies"]]
                else:
                    replies = value["replies"]
                self._span.update(output=replies)
            else:
                coerced_value = tracing_utils.coerce_tag_value(value)
                self._span.update(output=coerced_value)

        self._data[key] = value

    def raw_span(self) -> LangfuseClientSpan:
        """
        Return the underlying span instance.

        :return: The Langfuse span instance.
        """
        return self._span

    def get_data(self) -> Dict[str, Any]:
        """
        Return the data associated with the span.

        :return: The data associated with the span.
        """
        return self._data

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True)
class SpanContext:
    """
    Context for creating spans in Langfuse.

    Encapsulates the information needed to create and configure a span in Langfuse tracing.
    Used by SpanHandler to determine the span type (trace, generation, or default) and its configuration.

    :param name: The name of the span to create. For components, this is typically the component name.
    :param operation_name: The operation being traced (e.g. "haystack.pipeline.run"). Used to determine
        if a new trace should be created without warning.
    :param component_type: The type of component creating the span (e.g. "OpenAIChatGenerator").
        Can be used to determine the type of span to create.
    :param tags: Additional metadata to attach to the span. Contains component input/output data
        and other trace information.
    :param parent_span: The parent span if this is a child span. If None, a new trace will be created.
    :param trace_name: The name to use for the trace when creating a parent span. Defaults to "Haystack".
    :param public: Whether traces should be publicly accessible. Defaults to False.
    """

    name: str
    operation_name: str
    component_type: Optional[str]
    tags: Dict[str, Any]
    parent_span: Optional[Span]
    trace_name: str = "Haystack"
    public: bool = False

    def __post_init__(self) -> None:
        """
        Validate the span context attributes.

        :raises ValueError: If name, operation_name or trace_name are empty
        :raises TypeError: If tags is not a dictionary
        """
        if not self.name:
            msg = "Span name cannot be empty"
            raise ValueError(msg)
        if not self.operation_name:
            msg = "Operation name cannot be empty"
            raise ValueError(msg)
        if not self.trace_name:
            msg = "Trace name cannot be empty"
            raise ValueError(msg)


class SpanHandler(ABC):
    """
    Abstract base class for customizing how Langfuse spans are created and processed.

    This class defines two key extension points:
    1. create_span: Controls what type of span to create (default or generation)
    2. handle: Processes the span after component execution (adding metadata, metrics, etc.)

    To implement a custom handler:
    - Extend this class or DefaultSpanHandler
    - Override create_span and handle methods. It is more common to override handle.
    - Pass your handler to LangfuseConnector init method
    """

    def __init__(self) -> None:
        self.tracer: Optional[langfuse.Langfuse] = None

    def init_tracer(self, tracer: langfuse.Langfuse) -> None:
        """
        Initialize with Langfuse tracer. Called internally by LangfuseTracer.

        :param tracer: The Langfuse client instance to use for creating spans
        """
        self.tracer = tracer

    @abstractmethod
    def create_span(self, context: SpanContext) -> LangfuseSpan:
        """
        Create a span of appropriate type based on the context.

        This method determines what kind of span to create:
        - A new trace if there's no parent span
        - A generation span for LLM components
        - A default span for other components

        :param context: The context containing all information needed to create the span
        :returns: A new LangfuseSpan instance configured according to the context
        """
        pass

    @abstractmethod
    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        """
        Process a span after component execution by attaching metadata and metrics.

        This method is called after the component or pipeline yields its span, allowing you to:
        - Extract and attach token usage statistics
        - Add model information
        - Record timing data (e.g., time-to-first-token)
        - Set log levels for quality monitoring
        - Add custom metrics and observations

        :param span: The span that was yielded by the component
        :param component_type: The type of component that created the span, used to determine
            what metadata to extract and how to process it
        """
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanHandler":
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)


def _sanitize_usage_data(usage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize usage data for Langfuse by flattening to a single-level dictionary.

    Langfuse's usage_details must be a flat dictionary with only numeric values. This function:
    - Flattens nested dictionaries using dot notation (e.g., cache_creation.input_tokens)
    - Keeps int and float values
    - Skips None, boolean, string, and other non-numeric types

    :param usage: Raw usage dictionary from the provider.
    :returns: Flat dictionary with only numeric values (int or float).
    """
    if not isinstance(usage, dict):
        return {}

    sanitized: Dict[str, Any] = {}

    def _flatten(data: Dict[str, Any], prefix: str = "") -> None:
        """Recursively flatten nested dictionaries."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if value is None:
                # Skip None values (e.g., Anthropic's server_tool_use)
                continue
            elif isinstance(value, bool):
                # Skip boolean values
                continue
            elif isinstance(value, (int, float)):
                # Keep numeric values
                sanitized[full_key] = value
            elif isinstance(value, dict):
                # Recursively flatten nested dicts
                _flatten(value, full_key)
            # Skip strings and other non-numeric types (e.g., Anthropic's service_tier)

    _flatten(usage)
    return sanitized


class DefaultSpanHandler(SpanHandler):
    """DefaultSpanHandler provides the default Langfuse tracing behavior for Haystack."""

    def create_span(self, context: SpanContext) -> LangfuseSpan:
        if self.tracer is None:
            message = (
                "Tracer is not initialized. "
                "Make sure the environment variable HAYSTACK_CONTENT_TRACING_ENABLED is set to true before "
                "importing Haystack."
            )
            raise RuntimeError(message)

        # Get external tracing context for root trace creation (correlation metadata)
        tracing_ctx = tracing_context_var.get({})
        if not context.parent_span:
            root_span_type: Literal["agent", "span"] = (
                "agent" if context.operation_name == "haystack.agent.run" else "span"
            )
            # Create a new trace when there's no parent span
            span_context_manager = self.tracer.start_as_current_observation(
                name=context.trace_name, version=tracing_ctx.get("version"), as_type=root_span_type
            )

            # Create LangfuseSpan which will handle entering the context manager
            span = LangfuseSpan(span_context_manager)

            # Build trace metadata from context
            trace_metadata: TraceMetadata = {
                "name": context.trace_name,
                "user_id": tracing_ctx.get("user_id"),
                "session_id": tracing_ctx.get("session_id"),
                "version": tracing_ctx.get("version"),
                "metadata": None,
                "tags": tracing_ctx.get("tags"),
                "public": context.public,
                "release": None,
            }

            # Filter out None values and apply trace attributes
            trace_attrs = {k: v for k, v in trace_metadata.items() if v is not None}
            if trace_attrs:
                span._span.update_trace(**trace_attrs)

            return span

        span_type = None

        if context.component_type == "ToolInvoker":
            span_type = "tool"
        elif context.operation_name == "haystack.agent.run":
            span_type = "agent"
        elif context.component_type and context.component_type.endswith("Retriever"):
            span_type = "retriever"
        elif context.component_type and context.component_type.endswith("Embedder"):
            span_type = "embedding"
        elif context.component_type and context.component_type.endswith("Generator"):
            span_type = "generation"

        if span_type:
            return LangfuseSpan(
                self.tracer.start_as_current_observation(
                    name=context.name, as_type=cast(ObservationSpanType, span_type)
                )
            )
        else:
            return LangfuseSpan(self.tracer.start_as_current_span(name=context.name))

    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        # If the span is at the pipeline level, we add input and output keys to the span
        at_pipeline_level = span.get_data().get(_PIPELINE_INPUT_KEY) is not None
        if at_pipeline_level:
            coerced_input = tracing_utils.coerce_tag_value(span.get_data().get(_PIPELINE_INPUT_KEY))
            coerced_output = tracing_utils.coerce_tag_value(span.get_data().get(_PIPELINE_OUTPUT_KEY))
            span.raw_span().update_trace(input=coerced_input, output=coerced_output)
        # special case for ToolInvoker (to update the span name to be: `original_component_name - [tool_names]`)
        if component_type == "ToolInvoker":
            tool_names: List[str] = []
            messages = span.get_data().get(_COMPONENT_INPUT_KEY, {}).get("messages", [])
            for message in messages:
                if isinstance(message, ChatMessage) and message.tool_calls:
                    tool_names.extend(call.tool_name for call in message.tool_calls)

            if tool_names:
                # Fallback to "ToolInvoker" if we can't retrieve component name
                tool_invoker_name = span.get_data().get(_COMPONENT_NAME_KEY, "ToolInvoker")
                tool_counts = Counter(tool_names)  # how many times each tool was called
                formatted_names = [f"{name} (x{count})" if count > 1 else name for name, count in tool_counts.items()]
                span.raw_span().update(name=f"{tool_invoker_name} - {sorted(formatted_names)}")

        if component_type and component_type.endswith("ChatGenerator"):
            replies = span.get_data().get(_COMPONENT_OUTPUT_KEY, {}).get("replies")
            if replies:
                meta = replies[0].meta
                completion_start_time = meta.get("completion_start_time")
                if completion_start_time:
                    try:
                        completion_start_time = datetime.fromisoformat(completion_start_time)
                    except ValueError:
                        logger.error(f"Failed to parse completion_start_time: {completion_start_time}")
                        completion_start_time = None
                usage = meta.get("usage")
                sanitized_usage = _sanitize_usage_data(usage) if usage else None
                span.raw_span().update(
                    usage_details=sanitized_usage,
                    model=meta.get("model"),
                    completion_start_time=completion_start_time,
                )
        elif component_type and component_type.endswith("Generator"):
            meta = span.get_data().get(_COMPONENT_OUTPUT_KEY, {}).get("meta")
            if meta:
                usage = meta[0].get("usage")
                sanitized_usage = _sanitize_usage_data(usage) if usage else None
                span.raw_span().update(usage_details=sanitized_usage, model=meta[0].get("model"))


class LangfuseTracer(Tracer):
    """
    Internal class representing a bridge between the Haystack tracer and Langfuse.
    """

    def __init__(
        self,
        tracer: langfuse.Langfuse,
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
        # Keep _context as deprecated shim to avoid AttributeError if anyone uses it
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

        # Create a new span context
        span_context = SpanContext(
            name=span_name,
            operation_name=operation_name,
            component_type=component_type,
            tags=tags,
            # We use the current active span as the parent span if not provided to handle nested pipelines
            # The nested pipeline (or sub-pipeline) will be a child of the current active span
            parent_span=parent_span or self.current_span(),
            trace_name=self._name,
            public=self._public,
        )

        # Create span using the handler
        span = self._span_handler.create_span(span_context)

        # Build new span hierarchy: copy existing stack, add new span, save for restoration
        prev_stack = span_stack_var.get()
        new_stack = (prev_stack or []).copy()
        new_stack.append(span)
        token = span_stack_var.set(new_stack)

        span.set_tags(tags)

        try:
            yield span
        finally:
            # Always clean up context, even if nested operations fail
            try:
                # Process span data (may fail with nested pipeline exceptions)
                self._span_handler.handle(span, component_type)

                # End span (may fail if span data is corrupted)
                raw_span = span.raw_span()
                # In v3, we need to properly exit context managers
                if span._context_manager is not None:
                    span._context_manager.__exit__(None, None, None)
                elif hasattr(raw_span, "end"):
                    # Only call end() if it's not a context manager
                    raw_span.end()
            except Exception as cleanup_error:
                # Log cleanup errors but don't let them corrupt context
                logger.warning(
                    "Error during span cleanup for {operation_name}: {cleanup_error}",
                    operation_name=operation_name,
                    cleanup_error=cleanup_error,
                )
            finally:
                # Restore previous span stack using saved token - ensures proper cleanup
                span_stack_var.reset(token)

            if self.enforce_flush:
                self.flush()

    def flush(self) -> None:
        self._tracer.flush()

    def current_span(self) -> Optional[Span]:
        """
        Return the current active span.

        :return: The current span if available, else None.
        """
        # Get top of span stack (most recent span) from context-local storage
        stack = span_stack_var.get()
        return stack[-1] if stack else None

    def get_trace_url(self) -> str:
        """
        Return the URL to the tracing data.
        :return: The URL to the tracing data.
        """
        return self._tracer.get_trace_url() or ""

    def get_trace_id(self) -> str:
        """
        Return the trace ID.
        :return: The trace ID.
        """
        return self._tracer.get_current_trace_id() or ""
