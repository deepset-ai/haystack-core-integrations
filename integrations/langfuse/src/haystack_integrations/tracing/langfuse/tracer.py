import contextlib
import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union
from typing_extensions import TypeAlias

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer
from haystack.tracing import tracer as proxy_tracer
from haystack.tracing import utils as tracing_utils
import langfuse
from langfuse.client import (
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)

# Type alias for Langfuse stateful clients
LangfuseStatefulClient: TypeAlias = Union[
    StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
]

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

# Keys used by Haystack for traces and spans.
_PIPELINE_RUN_KEY = "haystack.pipeline.run"
_COMPONENT_NAME_KEY = "haystack.component.name"
_COMPONENT_TYPE_KEY = "haystack.component.type"
_COMPONENT_OUTPUT_KEY = "haystack.component.output"

# Context var used to keep track of tracing-related info.
tracing_context_var: ContextVar[Dict[Any, Any]] = ContextVar("tracing_context")


class LangfuseSpan(Span):
    """
    Internal class representing a bridge between the Haystack span tracing API and Langfuse.
    """

    def __init__(self, span: LangfuseStatefulClient) -> None:
        self._span = span
        self._data: Dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.update(metadata={key: coerced_value})
        self._data[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
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

    def raw_span(self) -> LangfuseStatefulClient:
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True)
class SpanContext:
    name: str
    operation_name: str
    component_type: Optional[str]
    tags: Dict[str, Any]
    parent_span: Optional[Span]
    trace_name: str = "Haystack"
    public: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Span name cannot be empty")
        if not self.operation_name:
            raise ValueError("Operation name cannot be empty")
        if not self.trace_name:
            raise ValueError("Trace name cannot be empty")


class SpanHandler(ABC):
    def __init__(self):
        self.tracer: Optional[langfuse.Langfuse] = None

    def init_tracer(self, tracer: langfuse.Langfuse) -> None:
        self.tracer = tracer

    @abstractmethod
    def create_span(self, context: SpanContext) -> LangfuseSpan:
        pass

    @abstractmethod
    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanHandler":
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)


class DefaultSpanHandler(SpanHandler):
    def create_span(self, context: SpanContext) -> LangfuseSpan:
        if self.tracer is None:
            raise RuntimeError("Tracer is not initialized")
        tracing_ctx = tracing_context_var.get({})
        if not context.parent_span:
            if context.operation_name != _PIPELINE_RUN_KEY:
                logger.warning(
                    "Creating a new trace without a parent span is not recommended for operation '{operation_name}'.",
                    operation_name=context.operation_name,
                )
            return LangfuseSpan(
                self.tracer.trace(
                    name=context.trace_name,
                    public=context.public,
                    id=tracing_ctx.get("trace_id"),
                    user_id=tracing_ctx.get("user_id"),
                    session_id=tracing_ctx.get("session_id"),
                    tags=tracing_ctx.get("tags"),
                    version=tracing_ctx.get("version"),
                )
            )
        elif context.component_type in _ALL_SUPPORTED_GENERATORS:
            return LangfuseSpan(
                context.parent_span.raw_span().generation(name=context.name)
            )
        else:
            return LangfuseSpan(context.parent_span.raw_span().span(name=context.name))

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
                        completion_start_time = datetime.fromisoformat(
                            completion_start_time
                        )
                    except ValueError:
                        logger.error(
                            f"Failed to parse completion_start_time: {completion_start_time}"
                        )
                        completion_start_time = None
                span._span.update(
                    usage=meta.get("usage") or None,
                    model=meta.get("model"),
                    completion_start_time=completion_start_time,
                )


class LangfuseTracer(Tracer):
    """
    Bridge between the Haystack tracer and Langfuse.
    """

    def __init__(
        self,
        tracer: langfuse.Langfuse,
        name: str = "Haystack",
        public: bool = False,
        span_handler: Optional[SpanHandler] = None,
        input_operation_name: Optional[str] = None,
        output_operation_name: Optional[str] = None,
    ) -> None:
        if not proxy_tracer.is_content_tracing_enabled:
            logger.warning(
                "Traces will not be logged to Langfuse because Haystack tracing is disabled. "
                "To enable, set the HAYSTACK_CONTENT_TRACING_ENABLED environment variable to true before importing Haystack."
            )
        self._tracer = tracer
        self._context: List[LangfuseSpan] = []
        self._name = name
        self._public = public
        self.enforce_flush = (
            os.getenv(HAYSTACK_LANGFUSE_ENFORCE_FLUSH_ENV_VAR, "true").lower() == "true"
        )
        self._span_handler = span_handler or DefaultSpanHandler()
        self._span_handler.init_tracer(tracer)
        self._input_operation_name = input_operation_name
        self._output_operation_name = output_operation_name
        self.input_value = None
        self.output_value = None

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Iterator[Span]:
        tags = tags or {}
        span_name = tags.get(_COMPONENT_NAME_KEY, operation_name)
        component_type = tags.get(_COMPONENT_TYPE_KEY)
        span = self._span_handler.create_span(
            SpanContext(
                name=span_name,
                operation_name=operation_name,
                component_type=component_type,
                tags=tags,
                parent_span=parent_span,
                trace_name=self._name,
                public=self._public,
            )
        )
        self._context.append(span)
        span.set_tags(tags)
        try:
            yield span
        finally:
            # Capture input/output if this span matches the designated names.
            if self._input_operation_name and span_name == self._input_operation_name:
                self.input_value = span._data.get("haystack.component.input", {}).get(
                    "query"
                )
            elif (
                self._output_operation_name and span_name == self._output_operation_name
            ):
                replies = span._data.get("haystack.component.output", {}).get("replies")
                if replies and len(replies) > 0:
                    self.output_value = replies[0]
            # For root spans (no parent), update with captured input/output values.
            if parent_span is None:
                span.raw_span().update(input=self.input_value, output=self.output_value)
            self._span_handler.handle(span, component_type)
            raw_span = span.raw_span()
            if isinstance(raw_span, (StatefulSpanClient, StatefulGenerationClient)):
                raw_span.end()
            self._context.pop()
            if self.enforce_flush:
                self.flush()

    def flush(self):
        self._tracer.flush()

    def current_span(self) -> Optional[Span]:
        return self._context[-1] if self._context else None

    def get_trace_url(self) -> str:
        return self._tracer.get_trace_url()