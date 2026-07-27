# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.tracing.rhesis.mapping import (
    APPENDIX_A_HAYSTACK_TAGS,
    APPENDIX_A_OPERATIONS,
    HAYSTACK_OPERATION_MAPPING,
    HAYSTACK_TAG_MAPPING,
    resolve_operation_type,
    resolve_span_name,
    sanitize_function_span_name,
)


class TestMappingCompleteness:
    def test_every_appendix_a_tag_has_mapping(self):
        for tag in APPENDIX_A_HAYSTACK_TAGS:
            assert tag in HAYSTACK_TAG_MAPPING, f"Missing mapping for {tag}"

    def test_every_appendix_a_operation_has_mapping(self):
        for operation in APPENDIX_A_OPERATIONS:
            assert operation in HAYSTACK_OPERATION_MAPPING, f"Missing mapping for {operation}"


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

    def test_generic_component_fallback(self):
        name = resolve_span_name(
            operation_name="haystack.component.run",
            component_type="DocumentJoiner",
            component_name="joiner",
            is_root=False,
        )
        assert name == sanitize_function_span_name("joiner")
