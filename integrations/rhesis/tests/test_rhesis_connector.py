# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from unittest.mock import Mock, patch

import pytest
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.utils import Secret

from haystack_integrations.components.connectors.rhesis import RhesisConnector
from haystack_integrations.tracing.rhesis import DefaultSpanHandler


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span, component_type=None):
        pass


class TestRhesisConnector:
    def test_run(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
            connector = RhesisConnector(name="Chat example")
            mock_tracer = Mock()
            mock_tracer.get_trace_url.return_value = "http://localhost:3000/traces?open_trace=abc"
            mock_tracer.get_trace_id.return_value = "abc123"
            connector.tracer = mock_tracer

            response = connector.run(invocation_context={"session_id": "sess-1"})
            assert response["name"] == "Chat example"
            assert response["trace_url"] == "http://localhost:3000/traces?open_trace=abc"
            assert response["trace_id"] == "abc123"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
            connector = RhesisConnector(name="Chat example")
            serialized = connector.to_dict()

        assert serialized == {
            "type": "haystack_integrations.components.connectors.rhesis.rhesis_connector.RhesisConnector",
            "init_parameters": {
                "name": "Chat example",
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["RHESIS_API_KEY"],
                    "strict": True,
                },
                "base_url": "http://localhost:8080",
                "project_id": None,
                "environment": "development",
                "frontend_url": None,
                "span_handler": None,
            },
        }

    def test_to_dict_with_custom_handler(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
            connector = RhesisConnector(name="Chat example", span_handler=CustomSpanHandler())
            serialized = connector.to_dict()

        assert serialized["init_parameters"]["span_handler"]["type"].endswith("CustomSpanHandler")

    def test_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
            connector = RhesisConnector(
                name="Chat example",
                base_url="http://localhost:8080",
                project_id="proj",
                environment="staging",
                frontend_url="http://localhost:3000",
            )
            data = connector.to_dict()
            restored = RhesisConnector.from_dict(data)
            assert restored.name == connector.name
            assert restored.base_url == connector.base_url
            assert restored.project_id == connector.project_id
            assert restored.environment == connector.environment
            assert restored.frontend_url == connector.frontend_url

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("RHESIS_API_KEY", raising=False)
        with (
            patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"),
            pytest.raises(Exception),
        ):
            RhesisConnector(name="Chat example", api_key=Secret.from_env_var("RHESIS_API_KEY"))

    def test_pipeline_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"):
            pipe = Pipeline()
            pipe.add_component("tracer", RhesisConnector("Chat example"))
            pipe.add_component("prompt_builder", ChatPromptBuilder())
            yaml = pipe.dumps()
            restored = Pipeline.loads(yaml)
            tracer = restored.get_component("tracer")
            assert tracer.name == "Chat example"
            assert "token-value" not in yaml
            assert "test-key" not in yaml

    def test_enable_tracing_called(self, monkeypatch):
        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        with (
            patch("haystack_integrations.components.connectors.rhesis.rhesis_connector.get_tracer_provider"),
            patch(
                "haystack_integrations.components.connectors.rhesis.rhesis_connector.tracing.enable_tracing"
            ) as mock_enable,
        ):
            RhesisConnector(name="Chat example")
            mock_enable.assert_called_once()
