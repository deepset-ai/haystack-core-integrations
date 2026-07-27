# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack-to-Rhesis semantic mapping tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rhesis.sdk.telemetry.attributes import AIAttributes
from rhesis.telemetry.constants import ConversationContext, TestExecutionContext
from rhesis.telemetry.schemas import AIOperationType

_PIPELINE_RUN_KEY = "haystack.pipeline.run"
_ASYNC_PIPELINE_RUN_KEY = "haystack.async_pipeline.run"
_AGENT_RUN_KEY = "haystack.agent.run"
_COMPONENT_RUN_KEY = "haystack.component.run"
_COMPONENT_NAME_KEY = "haystack.component.name"
_COMPONENT_TYPE_KEY = "haystack.component.type"

# Every Haystack tag emitted per Appendix A (haystack 2.30.x). Used by completeness tests.
APPENDIX_A_HAYSTACK_TAGS: frozenset[str] = frozenset(
    {
        "haystack.pipeline.input_data",
        "haystack.pipeline.output_data",
        "haystack.pipeline.metadata",
        "haystack.pipeline.max_runs_per_component",
        "haystack.component.name",
        "haystack.component.type",
        "haystack.component.fully_qualified_type",
        "haystack.component.input_types",
        "haystack.component.input_spec",
        "haystack.component.output_spec",
        "haystack.component.input",
        "haystack.component.output",
        "haystack.agent.input",
        "haystack.agent.output",
    }
)

APPENDIX_A_OPERATIONS: frozenset[str] = frozenset(
    {
        _PIPELINE_RUN_KEY,
        _ASYNC_PIPELINE_RUN_KEY,
        _COMPONENT_RUN_KEY,
        _AGENT_RUN_KEY,
    }
)


class MappingPromotion(str, Enum):
    FIRST_CLASS = "first-class"
    METADATA = "metadata"
    MIXED = "mixed"


@dataclass(frozen=True)
class MappingTarget:
    """Describes how a Haystack tag maps onto Rhesis semantics."""

    rhesis_target: str
    promotion: MappingPromotion
    notes: str = ""


# Authoritative tag mapping (also summarized in README.md).
HAYSTACK_TAG_MAPPING: dict[str, MappingTarget] = {
    "haystack.pipeline.input_data": MappingTarget(
        ConversationContext.SpanAttributes.CONVERSATION_INPUT,
        MappingPromotion.FIRST_CLASS,
        "Promoted on root span at handler phase",
    ),
    "haystack.pipeline.output_data": MappingTarget(
        ConversationContext.SpanAttributes.CONVERSATION_OUTPUT,
        MappingPromotion.FIRST_CLASS,
        "Promoted on root span at handler phase",
    ),
    "haystack.pipeline.metadata": MappingTarget(
        "haystack.pipeline.metadata",
        MappingPromotion.METADATA,
    ),
    "haystack.pipeline.max_runs_per_component": MappingTarget(
        "haystack.pipeline.max_runs_per_component",
        MappingPromotion.METADATA,
    ),
    "haystack.component.name": MappingTarget(
        _COMPONENT_NAME_KEY,
        MappingPromotion.MIXED,
        "Used as OTel span name and metadata",
    ),
    "haystack.component.type": MappingTarget(
        "haystack.component.type",
        MappingPromotion.METADATA,
    ),
    "haystack.component.fully_qualified_type": MappingTarget(
        "haystack.component.fully_qualified_type",
        MappingPromotion.METADATA,
    ),
    "haystack.component.input_types": MappingTarget(
        "haystack.component.input_types",
        MappingPromotion.METADATA,
    ),
    "haystack.component.input_spec": MappingTarget(
        "haystack.component.input_spec",
        MappingPromotion.METADATA,
    ),
    "haystack.component.output_spec": MappingTarget(
        "haystack.component.output_spec",
        MappingPromotion.METADATA,
    ),
    "haystack.component.input": MappingTarget(
        "ai.prompt events / content attributes",
        MappingPromotion.MIXED,
        "Content-gated via HAYSTACK_CONTENT_TRACING_ENABLED",
    ),
    "haystack.component.output": MappingTarget(
        "ai.completion events / content attributes",
        MappingPromotion.MIXED,
        "Content-gated via HAYSTACK_CONTENT_TRACING_ENABLED",
    ),
    "haystack.agent.input": MappingTarget(
        AIAttributes.AGENT_INPUT_CONTENT,
        MappingPromotion.FIRST_CLASS,
    ),
    "haystack.agent.output": MappingTarget(
        AIAttributes.AGENT_OUTPUT_CONTENT,
        MappingPromotion.FIRST_CLASS,
    ),
}

# Operation-level mapping for span naming / kinds.
HAYSTACK_OPERATION_MAPPING: dict[str, MappingTarget] = {
    _PIPELINE_RUN_KEY: MappingTarget(
        "function.haystack.pipeline.run",
        MappingPromotion.FIRST_CLASS,
        "pipeline forbidden in ai.* namespace",
    ),
    _ASYNC_PIPELINE_RUN_KEY: MappingTarget(
        "function.haystack.async_pipeline.run",
        MappingPromotion.FIRST_CLASS,
    ),
    _AGENT_RUN_KEY: MappingTarget(
        AIOperationType.AGENT_INVOKE,
        MappingPromotion.FIRST_CLASS,
        "ai.operation.type=agent.invoke",
    ),
    _COMPONENT_RUN_KEY: MappingTarget(
        "resolved per component type",
        MappingPromotion.MIXED,
        "See SPAN_KIND_RULES",
    ),
}

# Ordered span-kind rules (first match wins). Unit-tested independently of inline conditionals.
SPAN_KIND_RULES: tuple[tuple[str, str, str], ...] = (
    # (operation_name, component_type predicate suffix or exact, rhesis span name)
    (_AGENT_RUN_KEY, "__root__", AIOperationType.AGENT_INVOKE),
    (_PIPELINE_RUN_KEY, "__root__", "function.haystack.pipeline.run"),
    (_ASYNC_PIPELINE_RUN_KEY, "__root__", "function.haystack.async_pipeline.run"),
    (_COMPONENT_RUN_KEY, "ToolInvoker", AIOperationType.TOOL_INVOKE),
    (_AGENT_RUN_KEY, "__any__", AIOperationType.AGENT_INVOKE),
    (_COMPONENT_RUN_KEY, "Retriever", AIOperationType.RETRIEVAL),
    (_COMPONENT_RUN_KEY, "Embedder", AIOperationType.EMBEDDING_GENERATE),
    (_COMPONENT_RUN_KEY, "Generator", AIOperationType.LLM_INVOKE),
)

_OPERATION_TYPE_BY_SPAN_NAME: dict[str, str] = {
    AIOperationType.LLM_INVOKE: AIAttributes.OPERATION_LLM_INVOKE,
    AIOperationType.TOOL_INVOKE: AIAttributes.OPERATION_TOOL_INVOKE,
    AIOperationType.RETRIEVAL: AIAttributes.OPERATION_RETRIEVAL,
    AIOperationType.EMBEDDING_GENERATE: AIAttributes.OPERATION_EMBEDDING_CREATE,
    AIOperationType.AGENT_INVOKE: AIAttributes.OPERATION_AGENT_INVOKE,
}

_INVOCATION_CONTEXT_FIELD_MAP: dict[str, str] = {
    "session_id": ConversationContext.SpanAttributes.CONVERSATION_ID,
    "conversation_id": ConversationContext.SpanAttributes.CONVERSATION_ID,
    "test_run_id": TestExecutionContext.SpanAttributes.TEST_RUN_ID,
    "test_id": TestExecutionContext.SpanAttributes.TEST_ID,
    "test_result_id": TestExecutionContext.SpanAttributes.TEST_RESULT_ID,
    "test_configuration_id": TestExecutionContext.SpanAttributes.TEST_CONFIGURATION_ID,
}


def sanitize_function_span_name(component_name: str) -> str:
    """Build a valid ``function.haystack.*`` span name for generic components."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", component_name).strip("_").lower()
    if not sanitized:
        sanitized = "component"
    return f"function.haystack.{sanitized}"


def resolve_span_name(
    *,
    operation_name: str,
    component_type: str | None,
    component_name: str,
    is_root: bool,
) -> str:
    """Return the Rhesis-compliant OTel span name for a Haystack operation."""
    if is_root:
        if operation_name == _AGENT_RUN_KEY:
            return AIOperationType.AGENT_INVOKE
        if operation_name == _PIPELINE_RUN_KEY:
            return "function.haystack.pipeline.run"
        if operation_name == _ASYNC_PIPELINE_RUN_KEY:
            return "function.haystack.async_pipeline.run"

    for rule_op, rule_type, span_name in SPAN_KIND_RULES:
        if rule_op != operation_name:
            continue
        if rule_type == "__root__":
            continue
        if rule_type == "__any__":
            return span_name
        if component_type and component_type.endswith(rule_type):
            return span_name
        if component_type == rule_type:
            return span_name

    return sanitize_function_span_name(component_name)


def resolve_operation_type(span_name: str) -> str | None:
    """Return ``ai.operation.type`` value for a Rhesis span name, if applicable."""
    return _OPERATION_TYPE_BY_SPAN_NAME.get(span_name)


def map_invocation_context(context: dict[str, Any]) -> dict[str, Any]:
    """Translate connector invocation_context keys to Rhesis span attributes."""
    attrs: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        mapped = _INVOCATION_CONTEXT_FIELD_MAP.get(key)
        if mapped:
            attrs[mapped] = value
        else:
            attrs[f"haystack.invocation.{key}"] = value

    session = context.get("session_id") or context.get("conversation_id")
    if session:
        attrs[AIAttributes.SESSION_ID] = session
        attrs[ConversationContext.SpanAttributes.IS_TURN_ROOT] = True
    return attrs
