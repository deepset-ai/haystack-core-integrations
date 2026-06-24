# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import tracing

from haystack_integrations.components.connectors.opentelemetry import OpenTelemetryConnector
from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer


@pytest.fixture(autouse=True)
def reset_tracing():
    # Each test enables global tracing; make sure we reset it afterwards so the state does not leak.
    yield
    tracing.disable_tracing()


class TestOpenTelemetryConnector:
    def test_init_enables_tracing(self) -> None:
        connector = OpenTelemetryConnector("test_connector")

        assert connector.name == "test_connector"
        assert isinstance(connector.tracer, OpenTelemetryTracer)
        assert tracing.is_tracing_enabled()
        assert tracing.tracer.actual_tracer is connector.tracer

    def test_default_name(self) -> None:
        connector = OpenTelemetryConnector()
        assert connector.name == "opentelemetry"

    def test_run(self) -> None:
        connector = OpenTelemetryConnector("test_connector")
        result = connector.run()
        assert result == {"name": "test_connector"}

    def test_to_dict(self) -> None:
        connector = OpenTelemetryConnector("test_connector")
        data = connector.to_dict()
        assert data == {
            "type": "haystack_integrations.components.connectors.opentelemetry.opentelemetry_connector."
            "OpenTelemetryConnector",
            "init_parameters": {"name": "test_connector"},
        }

    def test_from_dict(self) -> None:
        data = {
            "type": "haystack_integrations.components.connectors.opentelemetry.opentelemetry_connector."
            "OpenTelemetryConnector",
            "init_parameters": {"name": "test_connector"},
        }
        connector = OpenTelemetryConnector.from_dict(data)
        assert connector.name == "test_connector"
        assert isinstance(connector.tracer, OpenTelemetryTracer)
