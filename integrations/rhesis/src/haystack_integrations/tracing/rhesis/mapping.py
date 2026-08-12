# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack-to-Rhesis semantic mapping tables."""

from __future__ import annotations

import re
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

# Haystack 3.0 moved the Agent loop off ``Pipeline._run_component``: each iteration now opens its own
# span, and the LLM call and every tool call are traced directly instead of through a ``ToolInvoker``
# component span. These operations carry no ``haystack.component.name``/``.type`` tags, so they are
# matched on operation name alone (see ``_OPERATION_ONLY``).
_AGENT_STEP_KEY = "haystack.agent.step"
_AGENT_STEP_LLM_KEY = "haystack.agent.step.llm"
_AGENT_STEP_TOOL_KEY = "haystack.agent.step.tool"
_TOOL_NAME_KEY = "haystack.tool.name"
_TOOL_DESCRIPTION_KEY = "haystack.tool.description"

AGENT_STEP_SPAN_NAME = "function.haystack.agent.step"

# The tables below hold `AIOperationType.<X>.value`, never the member. `AIOperationType` is a
# `(str, Enum)` rather than a `StrEnum`, so `str(member)` renders as "AIOperationType.LLM_INVOKE".
# Comparisons, dict lookups and the OTLP encoder all resolve to the value, which is why this went
# unnoticed — but anything that formats a span name into text gets the enum repr instead.

# Sentinel for rules keyed on operation name alone, used by the haystack 3.0 agent-loop spans which
# carry no component tags at all.
_OPERATION_ONLY = "__operation__"

# A trace root is named by its operation alone, whatever component happens to be running. Kept
# separate from SPAN_KIND_RULES because these three win over every rule below, and expressing that
# in an ordered first-match table needed rows the matcher then had to skip.
ROOT_SPAN_NAMES: dict[str, str] = {
    _AGENT_RUN_KEY: AIOperationType.AGENT_INVOKE.value,
    _PIPELINE_RUN_KEY: "function.haystack.pipeline.run",
    _ASYNC_PIPELINE_RUN_KEY: "function.haystack.async_pipeline.run",
}

# Ordered span-kind rules (first match wins). Unit-tested independently of inline conditionals.
# Root spans are named by ROOT_SPAN_NAMES above, not from here.
SPAN_KIND_RULES: tuple[tuple[str, str, str], ...] = (
    # (operation_name, component_type suffix, rhesis span name)
    (_AGENT_STEP_LLM_KEY, _OPERATION_ONLY, AIOperationType.LLM_INVOKE.value),
    (_AGENT_STEP_TOOL_KEY, _OPERATION_ONLY, AIOperationType.TOOL_INVOKE.value),
    (_AGENT_STEP_KEY, _OPERATION_ONLY, AGENT_STEP_SPAN_NAME),
    # haystack 2.x emitted tool calls as a ToolInvoker component span; kept for backwards compatibility.
    (_COMPONENT_RUN_KEY, "ToolInvoker", AIOperationType.TOOL_INVOKE.value),
    (_AGENT_RUN_KEY, "__any__", AIOperationType.AGENT_INVOKE.value),
    (_COMPONENT_RUN_KEY, "Retriever", AIOperationType.RETRIEVAL.value),
    (_COMPONENT_RUN_KEY, "Embedder", AIOperationType.EMBEDDING_GENERATE.value),
    (_COMPONENT_RUN_KEY, "Generator", AIOperationType.LLM_INVOKE.value),
)

_OPERATION_TYPE_BY_SPAN_NAME: dict[str, str] = {
    AIOperationType.LLM_INVOKE.value: AIAttributes.OPERATION_LLM_INVOKE,
    AIOperationType.TOOL_INVOKE.value: AIAttributes.OPERATION_TOOL_INVOKE,
    AIOperationType.RETRIEVAL.value: AIAttributes.OPERATION_RETRIEVAL,
    AIOperationType.EMBEDDING_GENERATE.value: AIAttributes.OPERATION_EMBEDDING_CREATE,
    AIOperationType.AGENT_INVOKE.value: AIAttributes.OPERATION_AGENT_INVOKE,
    AIOperationType.AGENT_HANDOFF.value: AIAttributes.OPERATION_AGENT_HANDOFF,
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
    if is_root and operation_name in ROOT_SPAN_NAMES:
        return ROOT_SPAN_NAMES[operation_name]

    for rule_op, rule_type, span_name in SPAN_KIND_RULES:
        if rule_op != operation_name:
            continue
        if rule_type in ("__any__", _OPERATION_ONLY):
            return span_name
        # `endswith` covers the exact match too, since a string ends with itself.
        if component_type and component_type.endswith(rule_type):
            return span_name

    return sanitize_function_span_name(component_name)


def resolve_operation_type(span_name: str) -> str | None:
    """Return ``ai.operation.type`` value for a Rhesis span name, if applicable."""
    return _OPERATION_TYPE_BY_SPAN_NAME.get(span_name)


def map_invocation_context(context: dict[str, Any]) -> dict[str, Any]:
    """
    Translate connector invocation_context keys to Rhesis span attributes.

    The mapped identifiers are stringified. They land on typed string columns in the Rhesis span
    schema, and the exporter validates a whole export batch at once — so one span carrying, say, an
    integer database row id as ``session_id`` would fail validation and take every other span in its
    batch down with it. Unmapped keys pass through unchanged into free-form attributes.
    """
    attrs: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        mapped = _INVOCATION_CONTEXT_FIELD_MAP.get(key)
        if mapped:
            attrs[mapped] = str(value)
        else:
            attrs[f"haystack.invocation.{key}"] = value

    session = context.get("session_id") or context.get("conversation_id")
    if session:
        attrs[AIAttributes.SESSION_ID] = str(session)
        attrs[ConversationContext.SpanAttributes.IS_TURN_ROOT] = True
    return attrs
