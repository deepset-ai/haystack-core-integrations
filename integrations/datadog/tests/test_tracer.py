# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ddtrace.trace import Span as ddSpan
from ddtrace.trace import tracer as dd_tracer

from haystack_integrations.tracing.datadog import DatadogTracer


@pytest.fixture(autouse=True)
def disable_agent_writer():
    # Disable the global Datadog tracer so that spans are still created (and remain inspectable) but are never
    # flushed to a Datadog agent over the network. This keeps the tests offline and avoids noisy connection errors.
    original_enabled = dd_tracer.enabled
    dd_tracer.enabled = False
    yield
    dd_tracer.enabled = original_enabled


class TestDatadogTracer:
    def test_trace_sets_component_resource_name(self) -> None:
        tracer = DatadogTracer(dd_tracer)

        component_tags = {
            "haystack.component.name": "test_component",
            "haystack.component.type": "TestType",
            "haystack.component.input": {"input_key": "input_value"},
            "haystack.component.output": {"output_key": "output_value"},
        }

        with tracer.trace("haystack.component.run", tags=component_tags) as span:
            span.set_tag("key", "value")
            raw_span = span.raw_span()

        assert raw_span.name == "haystack.component.run"
        assert "test_component" in raw_span.resource
        assert "TestType" in raw_span.resource

    def test_tagging(self) -> None:
        tracer = DatadogTracer(dd_tracer)

        with tracer.trace("test", tags={"key1": "value1"}) as span:
            span.set_tag("key2", "value2")
            raw_span = span.raw_span()

        assert raw_span.get_tag("key1") == "value1"
        assert raw_span.get_tag("key2") == "value2"

    def test_current_span(self) -> None:
        tracer = DatadogTracer(dd_tracer)

        with tracer.trace("test"):
            current_span = tracer.current_span()
            assert current_span is not None
            current_span.set_tag("key1", "value1")

            raw_span = current_span.raw_span()
            assert raw_span is not None
            assert isinstance(raw_span, ddSpan)

            raw_span.set_tag("key2", "value2")

        assert raw_span.get_tag("key1") == "value1"
        assert raw_span.get_tag("key2") == "value2"

    def test_tracing_complex_values(self) -> None:
        tracer = DatadogTracer(dd_tracer)

        with tracer.trace("test") as span:
            span.set_tag("key", {"a": 1, "b": [2, 3, 4]})
            raw_span = span.raw_span()

        assert raw_span.get_tag("key") == '{"a": 1, "b": [2, 3, 4]}'

    def test_get_log_correlation_info(self) -> None:
        tracer = DatadogTracer(dd_tracer)

        with tracer.trace("test") as span:
            span.set_tag("key", "value")

            correlation_data = span.get_correlation_data_for_logs()

        # Depending on the ddtrace version, the correlation fields are returned either with or without a "dd." prefix.
        normalized = {key.removeprefix("dd."): value for key, value in correlation_data.items()}
        for field in ["trace_id", "span_id", "service", "env", "version"]:
            assert field in normalized
            assert isinstance(normalized[field], str)
