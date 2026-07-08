# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import tracing

from haystack_integrations.components.connectors.datadog import DatadogConnector
from haystack_integrations.tracing.datadog import DatadogTracer


@pytest.fixture(autouse=True)
def reset_tracing():
    # Each test enables global tracing; make sure we reset it afterwards so the state does not leak.
    yield
    tracing.disable_tracing()


class TestDatadogConnector:
    def test_init_enables_tracing(self) -> None:
        connector = DatadogConnector("test_connector")

        assert connector.name == "test_connector"
        assert isinstance(connector.tracer, DatadogTracer)
        assert tracing.is_tracing_enabled()
        assert tracing.tracer.actual_tracer is connector.tracer

    def test_default_name(self) -> None:
        connector = DatadogConnector()
        assert connector.name == "datadog"

    def test_run(self) -> None:
        connector = DatadogConnector("test_connector")
        result = connector.run()
        assert result == {"name": "test_connector"}

    def test_to_dict(self) -> None:
        connector = DatadogConnector("test_connector")
        data = connector.to_dict()
        assert data == {
            "type": "haystack_integrations.components.connectors.datadog.datadog_connector.DatadogConnector",
            "init_parameters": {"name": "test_connector"},
        }

    def test_from_dict(self) -> None:
        data = {
            "type": "haystack_integrations.components.connectors.datadog.datadog_connector.DatadogConnector",
            "init_parameters": {"name": "test_connector"},
        }
        connector = DatadogConnector.from_dict(data)
        assert connector.name == "test_connector"
        assert isinstance(connector.tracer, DatadogTracer)
