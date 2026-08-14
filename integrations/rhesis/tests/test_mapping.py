# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from rhesis.telemetry.attributes import AIAttributes
from rhesis.telemetry.constants import ConversationContext, TestExecutionContext

from haystack_integrations.tracing.rhesis.mapping import (
    AGENT_STEP_SPAN_NAME,
    map_invocation_context,
    resolve_operation_type,
    resolve_span_name,
    sanitize_function_span_name,
)


class TestResolveSpanName:
    def test_pipeline_root(self):
        name = resolve_span_name(
            operation_name="haystack.pipeline.run",
            component_type=None,
            component_name="haystack.pipeline.run",
            is_root=True,
        )
        assert name == "function.haystack.pipeline.run"

    def test_async_pipeline_root(self):
        name = resolve_span_name(
            operation_name="haystack.async_pipeline.run",
            component_type=None,
            component_name="haystack.async_pipeline.run",
            is_root=True,
        )
        assert name == "function.haystack.async_pipeline.run"

    def test_agent_root(self):
        name = resolve_span_name(
            operation_name="haystack.agent.run",
            component_type=None,
            component_name="agent",
            is_root=True,
        )
        assert name == "ai.agent.invoke"

    def test_chat_generator_child(self):
        name = resolve_span_name(
            operation_name="haystack.component.run",
            component_type="OpenAIChatGenerator",
            component_name="llm",
            is_root=False,
        )
        assert name == "ai.llm.invoke"
        assert resolve_operation_type(name) == "llm.invoke"

    def test_tool_invoker_child_haystack_2x(self):
        name = resolve_span_name(
            operation_name="haystack.component.run",
            component_type="ToolInvoker",
            component_name="tools",
            is_root=False,
        )
        assert name == "ai.tool.invoke"
        assert resolve_operation_type(name) == "tool.invoke"

    def test_agent_step_matched_by_operation_alone(self):
        name = resolve_span_name(
            operation_name="haystack.agent.step",
            component_type=None,
            component_name="haystack.agent.step",
            is_root=False,
        )
        assert name == AGENT_STEP_SPAN_NAME

    def test_agent_step_llm_matched_by_operation_alone(self):
        name = resolve_span_name(
            operation_name="haystack.agent.step.llm",
            component_type=None,
            component_name="haystack.agent.step.llm",
            is_root=False,
        )
        assert name == "ai.llm.invoke"
        assert resolve_operation_type(name) == "llm.invoke"

    def test_agent_step_tool_matched_by_operation_alone(self):
        name = resolve_span_name(
            operation_name="haystack.agent.step.tool",
            component_type=None,
            component_name="haystack.agent.step.tool",
            is_root=False,
        )
        assert name == "ai.tool.invoke"
        assert resolve_operation_type(name) == "tool.invoke"

    def test_generic_component_fallback(self):
        name = resolve_span_name(
            operation_name="haystack.component.run",
            component_type="DocumentJoiner",
            component_name="joiner",
            is_root=False,
        )
        assert name == sanitize_function_span_name("joiner")


class TestMapInvocationContext:
    def test_known_keys_map_to_rhesis_attributes(self):
        attrs = map_invocation_context({"session_id": "sess-1", "test_run_id": "tr-1"})
        assert attrs[ConversationContext.SpanAttributes.CONVERSATION_ID] == "sess-1"
        assert attrs[TestExecutionContext.SpanAttributes.TEST_RUN_ID] == "tr-1"
        assert attrs[ConversationContext.SpanAttributes.IS_TURN_ROOT] is True

    def test_unknown_keys_fall_back_to_the_haystack_namespace(self):
        attrs = map_invocation_context({"tenant": "acme"})
        assert attrs["haystack.invocation.tenant"] == "acme"

    def test_none_values_are_dropped(self):
        assert map_invocation_context({"session_id": None}) == {}

    def test_non_string_ids_are_stringified(self):
        """
        A database row id or Slack user id passed as ``session_id`` must not poison the export.

        The mapped ids land on typed string columns and the exporter validates a whole batch at
        once, so one span carrying an int here would fail validation and take up to 511 valid spans
        with it.
        """
        attrs = map_invocation_context({"session_id": 12345, "test_id": 7})
        assert attrs[ConversationContext.SpanAttributes.CONVERSATION_ID] == "12345"
        assert attrs[TestExecutionContext.SpanAttributes.TEST_ID] == "7"
        assert attrs[AIAttributes.SESSION_ID] == "12345"
