# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Rhesis tracing bridge for Haystack.

Set ``HAYSTACK_CONTENT_TRACING_ENABLED=true`` before importing Haystack to capture
input/output content on spans.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from haystack import default_from_dict, default_to_dict
from haystack import logging as haystack_logging
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tracing import Span, Tracer
from haystack.tracing import tracer as proxy_tracer
from haystack.tracing import utils as tracing_utils
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode

from haystack_integrations.tracing.rhesis.mapping import (
    map_invocation_context,
    resolve_operation_type,
    resolve_span_name,
)
from rhesis.sdk.telemetry.attributes import AIAttributes, AIEvents
from rhesis.sdk.telemetry.utils.token_extraction import extract_token_usage
from rhesis.telemetry.constants import ConversationContext

logger = haystack_logging.getLogger(__name__)

HAYSTACK_RHESIS_ENFORCE_FLUSH_ENV_VAR = "HAYSTACK_RHESIS_ENFORCE_FLUSH"
MAX_CONTENT_LENGTH = 8000

# Root span plus the top-level agent span; anything deeper is a nested agent turn.
_OUTERMOST_AGENT_STACK_DEPTH = 2

_PIPELINE_INPUT_KEY = "haystack.pipeline.input_data"
_PIPELINE_OUTPUT_KEY = "haystack.pipeline.output_data"
_COMPONENT_NAME_KEY = "haystack.component.name"
_COMPONENT_TYPE_KEY = "haystack.component.type"
_COMPONENT_OUTPUT_KEY = "haystack.component.output"
_COMPONENT_INPUT_KEY = "haystack.component.input"
_AGENT_INPUT_KEY = "haystack.agent.input"
_AGENT_OUTPUT_KEY = "haystack.agent.output"
_AGENT_RUN_KEY = "haystack.agent.run"

# haystack 3.0 agent-loop operations. These spans carry none of the ``haystack.component.*`` tags,
# so they are recognised by operation name rather than component type.
_AGENT_STEP_LLM_KEY = "haystack.agent.step.llm"
_AGENT_STEP_TOOL_KEY = "haystack.agent.step.tool"
_AGENT_STEP_LLM_OUTPUT_KEY = "haystack.agent.step.llm.output"
_AGENT_STEP_TOOL_INPUT_KEY = "haystack.agent.step.tool.input"
_AGENT_STEP_TOOL_OUTPUT_KEY = "haystack.agent.step.tool.output"
_TOOL_NAME_KEY = "haystack.tool.name"

tracing_context_var: ContextVar[dict[Any, Any]] = ContextVar("rhesis_tracing_context")
span_stack_var: ContextVar[list[RhesisSpan] | None] = ContextVar("rhesis_span_stack", default=None)


@contextmanager
def rhesis_invocation_context(invocation_context: dict[str, Any] | None = None) -> Iterator[None]:
    """Attach Rhesis session/test metadata for the current async task or thread."""
    token = tracing_context_var.set(invocation_context or {})
    try:
        yield
    finally:
        tracing_context_var.reset(token)


def _message_text(message: Any) -> str:
    if isinstance(message, ChatMessage):
        return message.text or ""
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in parts if part)
        return str(content)
    return str(message)


def _messages_from_agent_payload(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        messages = payload.get("messages")
        return messages if isinstance(messages, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _extract_agent_conversation_io(data: dict[str, Any]) -> tuple[str, str]:
    """Return user input and assistant output text from agent span tags."""
    conv_input = ""
    for message in reversed(_messages_from_agent_payload(data.get(_AGENT_INPUT_KEY))):
        if isinstance(message, ChatMessage) and message.is_from(ChatRole.USER):
            conv_input = _message_text(message)
            break
        if isinstance(message, dict) and message.get("role") == "user":
            conv_input = _message_text(message)
            break

    conv_output = ""
    for message in reversed(_messages_from_agent_payload(data.get(_AGENT_OUTPUT_KEY))):
        if isinstance(message, ChatMessage) and message.is_from(ChatRole.ASSISTANT):
            conv_output = _message_text(message)
            break
        if isinstance(message, dict) and message.get("role") == "assistant":
            conv_output = _message_text(message)
            break

    return conv_input, conv_output


def _is_outermost_agent_turn_span() -> bool:
    """True for the top-level agent span in a hayhooks/pipeline request."""
    stack = span_stack_var.get() or []
    return len(stack) <= _OUTERMOST_AGENT_STACK_DEPTH


def _apply_conversation_turn_attributes(otel_span: trace.Span) -> None:
    tracing_ctx = tracing_context_var.get({})
    session = tracing_ctx.get("session_id") or tracing_ctx.get("conversation_id")
    if not session:
        return
    for key, value in map_invocation_context(tracing_ctx).items():
        otel_span.set_attribute(key, _coerce_attribute_value(value))


def _stamp_conversation_io(otel_span: trace.Span, conv_input: str, conv_output: str) -> None:
    if conv_input:
        otel_span.set_attribute(
            ConversationContext.SpanAttributes.CONVERSATION_INPUT,
            conv_input[: ConversationContext.MAX_IO_LENGTH],
        )
    if conv_output:
        otel_span.set_attribute(
            ConversationContext.SpanAttributes.CONVERSATION_OUTPUT,
            conv_output[: ConversationContext.MAX_IO_LENGTH],
        )


@dataclass
class RhesisTelemetry:
    """Thin wrapper around the OTel provider used by the Haystack integration."""

    provider: TracerProvider
    otel_tracer: trace.Tracer
    project_id: str | None
    environment: str
    base_url: str
    frontend_url: str | None = None

    def flush(self) -> None:
        """Flush pending spans to the Rhesis backend."""
        try:
            self.provider.force_flush(timeout_millis=30_000)
        except Exception as exc:
            logger.warning("Failed to flush Rhesis traces: {error}", error=exc)


def resolve_frontend_url(base_url: str, frontend_url: str | None) -> str:
    """Resolve the Rhesis frontend base URL for trace deep links."""
    if frontend_url:
        return frontend_url.rstrip("/")
    normalized = base_url.rstrip("/")
    if normalized in ("http://localhost:8080", "http://127.0.0.1:8080"):
        return "http://localhost:3000"
    if normalized == "https://api.rhesis.ai":
        return "https://app.rhesis.ai"
    return ""


def build_trace_url(frontend_url: str, trace_id: str, project_id: str | None) -> str:
    """Build a frontend deep link for the given trace."""
    if not frontend_url or not trace_id:
        return ""
    params: dict[str, str] = {"open_trace": trace_id}
    if project_id:
        params["project_id"] = project_id
    return f"{frontend_url}/traces?{urlencode(params)}"


class RhesisSpan(Span):
    """Bridge between Haystack's span API and OpenTelemetry spans for Rhesis."""

    def __init__(self, context_manager: AbstractContextManager[Any], operation_name: str = "") -> None:
        self._span = context_manager.__enter__()
        self._data: dict[str, Any] = {}
        self._context_manager = context_manager
        # Retained so post-run enrichment can tell haystack 3.0 agent-loop spans apart: they share the
        # same (absent) component type, so operation name is the only discriminator available.
        self.operation_name = operation_name

    def set_tag(self, key: str, value: Any) -> None:
        """Set a generic tag for this span."""
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.set_attribute(key, _coerce_attribute_value(coerced_value))
        self._data[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
        """Set a content-specific tag for this span when content tracing is enabled."""
        if not proxy_tracer.is_content_tracing_enabled:
            return

        self._data[key] = value

        if key.endswith(".input"):
            self._set_input_content(key, value)
        elif key.endswith(".output"):
            self._set_output_content(key, value)

    def _set_input_content(self, key: str, value: Any) -> None:
        if key == _AGENT_INPUT_KEY:
            text = _stringify_content(value)
            if text:
                self._span.set_attribute(AIAttributes.AGENT_INPUT_CONTENT, text[: ConversationContext.MAX_IO_LENGTH])
                self._span.add_event(
                    AIEvents.AGENT_INPUT, {AIAttributes.AGENT_INPUT_CONTENT: text[:MAX_CONTENT_LENGTH]}
                )
            return

        # Tool arguments are not a prompt, so they get the tool content attribute rather than
        # ai.prompt events.
        if key == _AGENT_STEP_TOOL_INPUT_KEY:
            text = _stringify_content(value)
            if text:
                self._span.set_attribute(AIAttributes.TOOL_INPUT_CONTENT, text[:MAX_CONTENT_LENGTH])
            return

        if isinstance(value, dict) and "messages" in value:
            messages = [m.to_openai_dict_format(require_tool_call_ids=False) for m in (value.get("messages") or [])]
            payload: Any
            if isinstance(gen_kwargs := value.get("generation_kwargs"), dict):
                payload = {"messages": messages, "generation_kwargs": gen_kwargs}
            else:
                payload = messages
            self._add_prompt_events(payload)
            return

        coerced_value = tracing_utils.coerce_tag_value(value)
        self._add_prompt_events(coerced_value)

    def _set_output_content(self, key: str, value: Any) -> None:
        if key == _AGENT_OUTPUT_KEY:
            text = _stringify_content(value)
            if text:
                self._span.set_attribute(AIAttributes.AGENT_OUTPUT_CONTENT, text[: ConversationContext.MAX_IO_LENGTH])
                self._span.add_event(
                    AIEvents.AGENT_OUTPUT,
                    {AIAttributes.AGENT_OUTPUT_CONTENT: text[:MAX_CONTENT_LENGTH]},
                )
            return

        if key == _AGENT_STEP_TOOL_OUTPUT_KEY:
            text = _stringify_content(value)
            if text:
                self._span.set_attribute(AIAttributes.TOOL_OUTPUT_CONTENT, text[:MAX_CONTENT_LENGTH])
            return

        if isinstance(value, dict) and "replies" in value:
            replies_list = value.get("replies") or []
            if all(isinstance(r, ChatMessage) for r in replies_list):
                replies = [m.to_openai_dict_format(require_tool_call_ids=False) for m in replies_list]
            else:
                replies = replies_list
            self._add_completion_events(replies)
            return

        coerced_value = tracing_utils.coerce_tag_value(value)
        self._add_completion_events(coerced_value)

    def _add_prompt_events(self, payload: Any) -> None:
        if isinstance(payload, dict) and "messages" in payload:
            for message in payload.get("messages") or []:
                if isinstance(message, dict):
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    self._span.add_event(
                        AIEvents.PROMPT,
                        {
                            AIAttributes.PROMPT_ROLE: role,
                            AIAttributes.PROMPT_CONTENT: str(content)[:MAX_CONTENT_LENGTH],
                        },
                    )
            return
        self._span.add_event(
            AIEvents.PROMPT,
            {AIAttributes.PROMPT_CONTENT: str(payload)[:MAX_CONTENT_LENGTH]},
        )

    def _add_completion_events(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    content = item.get("content", item)
                else:
                    content = item
                self._span.add_event(
                    AIEvents.COMPLETION,
                    {AIAttributes.COMPLETION_CONTENT: str(content)[:MAX_CONTENT_LENGTH]},
                )
            return
        self._span.add_event(
            AIEvents.COMPLETION,
            {AIAttributes.COMPLETION_CONTENT: str(payload)[:MAX_CONTENT_LENGTH]},
        )

    def raw_span(self) -> trace.Span:
        """Return the underlying OpenTelemetry span instance."""
        return self._span

    def get_data(self) -> dict[str, Any]:
        """Return the raw Haystack tag data collected for this span."""
        return self._data

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Return trace and span identifiers for log correlation."""
        try:
            ctx = self._span.get_span_context()
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }
        except (AttributeError, TypeError):
            return {}

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set multiple tags on this span."""
        for key, value in tags.items():
            self.set_tag(key, value)


@dataclass(frozen=True)
class SpanContext:
    """Context for creating spans in Rhesis."""

    name: str
    operation_name: str
    component_type: str | None
    tags: dict[str, Any]
    parent_span: Span | None
    trace_name: str = "Haystack"
    is_root: bool = False

    def __post_init__(self) -> None:
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
    """Extension point for customizing Rhesis span creation and enrichment."""

    def __init__(self) -> None:
        self.tracer: RhesisTelemetry | None = None

    def init_tracer(self, tracer: RhesisTelemetry) -> None:
        """Initialize with the Rhesis telemetry wrapper."""
        self.tracer = tracer

    @abstractmethod
    def create_span(self, context: SpanContext) -> RhesisSpan:
        """Create a span of appropriate type based on the context."""
        pass

    @abstractmethod
    def handle(self, span: RhesisSpan, component_type: str | None) -> None:
        """Process a span after component execution."""
        pass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanHandler:
        """Deserialize a SpanHandler from a dictionary."""
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this SpanHandler to a dictionary."""
        return default_to_dict(self)


def _coerce_attribute_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    coerced = tracing_utils.coerce_tag_value(value)
    if isinstance(coerced, str):
        return coerced
    try:
        return json.dumps(coerced, default=str)
    except (TypeError, ValueError):
        return str(coerced)


def _sanitize_usage_data(usage: dict[str, Any]) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}

    input_tokens, output_tokens, total_tokens = extract_token_usage(usage)
    sanitized: dict[str, int] = {}
    if input_tokens:
        sanitized[AIAttributes.LLM_TOKENS_INPUT] = input_tokens
    if output_tokens:
        sanitized[AIAttributes.LLM_TOKENS_OUTPUT] = output_tokens
    if total_tokens:
        sanitized[AIAttributes.LLM_TOKENS_TOTAL] = total_tokens
    return sanitized


def _set_token_attributes(span: trace.Span, usage: dict[str, Any] | None) -> None:
    if not usage:
        return
    for key, value in _sanitize_usage_data(usage).items():
        span.set_attribute(key, value)


def _apply_chat_reply_metadata(otel_span: trace.Span, replies: list[Any]) -> None:
    """Promote model name, timing, and token usage from the first chat reply's metadata."""
    meta = replies[0].meta
    completion_start_time = meta.get("completion_start_time")
    if completion_start_time:
        try:
            parsed = datetime.fromisoformat(completion_start_time)
            otel_span.set_attribute("haystack.llm.completion_start_time", parsed.isoformat())
        except ValueError:
            logger.error("Failed to parse completion_start_time: {value}", value=completion_start_time)
    model = meta.get("model")
    if model:
        otel_span.set_attribute(AIAttributes.MODEL_NAME, model)
    _set_token_attributes(otel_span, meta.get("usage"))


class DefaultSpanHandler(SpanHandler):
    """Default Rhesis tracing behavior for Haystack pipelines."""

    def create_span(self, context: SpanContext) -> RhesisSpan:
        """Create a Rhesis span based on the given Haystack context."""
        if self.tracer is None:
            msg = (
                "Tracer is not initialized. "
                "Make sure the environment variable HAYSTACK_CONTENT_TRACING_ENABLED is set to true "
                "before importing Haystack."
            )
            raise RuntimeError(msg)

        span_name = resolve_span_name(
            operation_name=context.operation_name,
            component_type=context.component_type,
            component_name=context.name,
            is_root=context.is_root,
        )

        parent_context = None
        if context.parent_span and isinstance(context.parent_span, RhesisSpan):
            parent_context = trace.set_span_in_context(context.parent_span.raw_span())

        span_cm = self.tracer.otel_tracer.start_as_current_span(
            name=span_name,
            context=parent_context,
            kind=trace.SpanKind.INTERNAL,
        )
        rhesis_span = RhesisSpan(span_cm, operation_name=context.operation_name)
        otel_span = rhesis_span.raw_span()

        operation_type = resolve_operation_type(span_name)
        if operation_type:
            otel_span.set_attribute(AIAttributes.OPERATION_TYPE, operation_type)

        if context.is_root:
            otel_span.set_attribute("haystack.trace.name", context.trace_name)
            tracing_ctx = tracing_context_var.get({})
            mapped = map_invocation_context(tracing_ctx)
            for key, value in mapped.items():
                otel_span.set_attribute(key, _coerce_attribute_value(value))

        # haystack 2.x routed tool calls through a ToolInvoker component span.
        if context.component_type == "ToolInvoker":
            otel_span.set_attribute(AIAttributes.TOOL_NAME, context.name)
            otel_span.set_attribute(AIAttributes.TOOL_TYPE, "haystack")

        # haystack 3.0 traces each tool call individually and names it in the span tags.
        if context.operation_name == _AGENT_STEP_TOOL_KEY:
            tool_name = context.tags.get(_TOOL_NAME_KEY)
            if tool_name:
                otel_span.set_attribute(AIAttributes.TOOL_NAME, str(tool_name))
            otel_span.set_attribute(AIAttributes.TOOL_TYPE, "haystack")

        return rhesis_span

    def handle(self, span: RhesisSpan, component_type: str | None) -> None:
        """Process and enrich a span after component execution."""
        otel_span = span.raw_span()
        data = span.get_data()

        conv_input = ""
        conv_output = ""
        if data.get(_PIPELINE_INPUT_KEY) is not None:
            conv_input = _stringify_content(data.get(_PIPELINE_INPUT_KEY))
            conv_output = _stringify_content(data.get(_PIPELINE_OUTPUT_KEY))
        elif data.get(_AGENT_INPUT_KEY) is not None and _is_outermost_agent_turn_span():
            conv_input, conv_output = _extract_agent_conversation_io(data)

        if conv_input or conv_output:
            _stamp_conversation_io(otel_span, conv_input, conv_output)
            if data.get(_PIPELINE_INPUT_KEY) is not None or _is_outermost_agent_turn_span():
                _apply_conversation_turn_attributes(otel_span)

        if component_type == "ToolInvoker":
            tool_names: list[str] = []
            messages = data.get(_COMPONENT_INPUT_KEY, {}).get("messages", [])
            for message in messages:
                if isinstance(message, ChatMessage) and message.tool_calls:
                    tool_names.extend(call.tool_name for call in message.tool_calls)
            if tool_names:
                tool_invoker_name = data.get(_COMPONENT_NAME_KEY, "ToolInvoker")
                tool_counts = Counter(tool_names)
                formatted_names = [
                    f"{name} (x{count})" if count > 1 else name for name, count in sorted(tool_counts.items())
                ]
                new_name = f"{tool_invoker_name} - [{', '.join(formatted_names)}]"
                if hasattr(otel_span, "update_name"):
                    otel_span.update_name(new_name)

        # In haystack 3.0 the agent calls its chat generator directly, so the reply metadata arrives on
        # the agent-loop LLM span instead of a ChatGenerator component span.
        if span.operation_name == _AGENT_STEP_LLM_KEY:
            llm_output = data.get(_AGENT_STEP_LLM_OUTPUT_KEY)
            replies = llm_output.get("replies") if isinstance(llm_output, dict) else None
            if replies:
                _apply_chat_reply_metadata(otel_span, replies)

        elif component_type and component_type.endswith("ChatGenerator"):
            replies = data.get(_COMPONENT_OUTPUT_KEY, {}).get("replies")
            if replies:
                _apply_chat_reply_metadata(otel_span, replies)

        elif component_type and component_type.endswith("Generator"):
            meta = data.get(_COMPONENT_OUTPUT_KEY, {}).get("meta")
            if meta:
                model = meta[0].get("model")
                if model:
                    otel_span.set_attribute(AIAttributes.MODEL_NAME, model)
                _set_token_attributes(otel_span, meta[0].get("usage"))

        elif component_type and component_type.endswith("Embedder"):
            output = data.get(_COMPONENT_OUTPUT_KEY, {})
            meta = output.get("meta")
            if meta and isinstance(meta, dict):
                usage = meta.get("usage") or meta.get("billed_units")
                _set_token_attributes(otel_span, usage)
                model = meta.get("model")
                if model and isinstance(model, str):
                    otel_span.set_attribute(AIAttributes.MODEL_NAME, model)


class RhesisTracer(Tracer):
    """Haystack tracer implementation that exports spans to Rhesis via OpenTelemetry."""

    def __init__(
        self,
        telemetry: RhesisTelemetry,
        name: str = "Haystack",
        span_handler: SpanHandler | None = None,
    ) -> None:
        """
        Initialize a RhesisTracer instance.

        :param telemetry: Configured Rhesis OpenTelemetry telemetry wrapper.
        :param name: Trace name shown in the Rhesis UI.
        :param span_handler: Custom handler for span creation and enrichment.
        """
        if not proxy_tracer.is_content_tracing_enabled:
            logger.warning(
                "Traces will not include input/output content in Rhesis because Haystack content tracing "
                "is disabled. To enable, set the HAYSTACK_CONTENT_TRACING_ENABLED environment variable "
                "to true before importing Haystack."
            )
        self._telemetry = telemetry
        self._name = name
        self._trace_id: str = ""
        self.enforce_flush = os.getenv(HAYSTACK_RHESIS_ENFORCE_FLUSH_ENV_VAR, "true").lower() == "true"
        self._span_handler = span_handler or DefaultSpanHandler()
        self._span_handler.init_tracer(telemetry)

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Create and manage a tracing span as a context manager."""
        tags = tags or {}
        span_name = tags.get(_COMPONENT_NAME_KEY, operation_name)
        component_type = tags.get(_COMPONENT_TYPE_KEY)
        current = parent_span or self.current_span()
        is_root_span = not (parent_span or self.current_span())

        span_context = SpanContext(
            name=span_name,
            operation_name=operation_name,
            component_type=component_type,
            tags=tags,
            parent_span=current,
            trace_name=self._name,
            is_root=is_root_span,
        )

        span = self._span_handler.create_span(span_context)

        prev_stack = span_stack_var.get()
        new_stack = (prev_stack or []).copy()
        new_stack.append(span)
        token = span_stack_var.set(new_stack)

        if is_root_span:
            try:
                raw_trace_id = span.raw_span().get_span_context().trace_id
                self._trace_id = format(raw_trace_id, "032x")
            except (AttributeError, TypeError):
                self._trace_id = ""

        span.set_tags(tags)

        try:
            yield span
        except Exception:
            exc_info = sys.exc_info()
            try:
                self._span_handler.handle(span, component_type)
                self._record_exception(span, exc_info)
                self._close_span(span, exc_info)
            except Exception as cleanup_error:
                logger.warning(
                    "Error during span cleanup for {operation_name}: {cleanup_error}",
                    operation_name=operation_name,
                    cleanup_error=cleanup_error,
                )
            raise
        else:
            try:
                self._span_handler.handle(span, component_type)
                self._close_span(span, None)
            except Exception as cleanup_error:
                logger.warning(
                    "Error during span cleanup for {operation_name}: {cleanup_error}",
                    operation_name=operation_name,
                    cleanup_error=cleanup_error,
                )
        finally:
            span_stack_var.reset(token)
            if self.enforce_flush:
                self.flush()

    def _record_exception(self, span: RhesisSpan, exc_info: tuple[Any, Any, Any]) -> None:
        exc_type, exc_value, _ = exc_info
        if exc_type is None:
            return
        otel_span = span.raw_span()
        otel_span.set_status(Status(StatusCode.ERROR, str(exc_value)))
        otel_span.record_exception(exc_value)
        otel_span.set_attribute(AIAttributes.ERROR_TYPE, exc_type.__name__)

    def _close_span(self, span: RhesisSpan, exc_info: tuple[Any, Any, Any] | None) -> None:
        if span._context_manager is not None:
            span._context_manager.__exit__(*exc_info if exc_info else (None, None, None))

    def flush(self) -> None:
        """Flush all pending spans to Rhesis."""
        self._telemetry.flush()

    def current_span(self) -> Span | None:
        """Return the current active span."""
        stack = span_stack_var.get()
        return stack[-1] if stack else None

    def get_trace_url(self) -> str:
        """Return the frontend URL for the current trace, when available."""
        frontend = resolve_frontend_url(self._telemetry.base_url, self._telemetry.frontend_url)
        return build_trace_url(frontend, self.get_trace_id(), self._telemetry.project_id)

    def get_trace_id(self) -> str:
        """Return the current trace ID."""
        return self._trace_id
