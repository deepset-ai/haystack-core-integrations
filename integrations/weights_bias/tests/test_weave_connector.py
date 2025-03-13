import os
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.tracing import tracer as haystack_configured_tracer
from weave.trace.autopatch import AutopatchSettings

from haystack_integrations.components.connectors import WeaveConnector
from haystack_integrations.tracing.weave import WeaveTracer


@component
class FailingComponent:
    """A component that always raises an exception for testing error handling."""

    def run(self) -> dict[str, Any]:
        """Execute the component's logic - always raises an exception."""
        msg = "Test error"
        raise ValueError(msg)


@pytest.fixture
def mock_weave_client() -> Generator[Mock, None, None]:
    mock_client = Mock()
    with patch("weave.init") as mock_init:
        mock_init.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_pipeline() -> Pipeline:
    builder_1 = PromptBuilder("Greeting: {{greeting}}")
    builder_2 = PromptBuilder("Predecessor said: {{predecessor}}")

    pp = Pipeline()
    pp.add_component("comp1", builder_1)
    pp.add_component("comp2", builder_2)
    pp.connect("comp1.prompt", "comp2.predecessor")

    return pp


class TestWeaveConnector:
    def test_serialization(self) -> None:
        """Test that WeaveConnector can be serialized and deserialized correctly"""
        connector = WeaveConnector(pipeline_name="test_pipeline")
        serialized: dict[str, Any] = connector.to_dict()

        assert serialized["init_parameters"]["pipeline_name"] == "test_pipeline"
        assert "type" in serialized
        assert serialized["type"] == "haystack_integrations.components.connectors.weave_connector.WeaveConnector"

        deserialized = WeaveConnector.from_dict(serialized)

        assert isinstance(deserialized, WeaveConnector)
        assert deserialized.pipeline_name == "test_pipeline"
        assert deserialized.tracer is None  # tracer is only initialized with warm_up
        assert deserialized.weave_init_kwargs == {}

    def test_serialization_of_weave_init_kwargs(self) -> None:
        """Test that WeaveConnector can be serialized and deserialized correctly"""
        connector = WeaveConnector(
            pipeline_name="test_pipeline",
            weave_init_kwargs={"autopatch_settings": AutopatchSettings(disable_autopatch=True)},
        )
        serialized: dict[str, Any] = connector.to_dict()

        assert serialized["init_parameters"]["pipeline_name"] == "test_pipeline"
        assert serialized["init_parameters"]["weave_init_kwargs"] == {"autopatch_settings": {"disable_autopatch": True}}
        assert "type" in serialized
        assert serialized["type"] == "haystack_integrations.components.connectors.weave_connector.WeaveConnector"

        deserialized = WeaveConnector.from_dict(serialized)

        assert isinstance(deserialized, WeaveConnector)
        assert deserialized.pipeline_name == "test_pipeline"
        assert deserialized.tracer is None  # tracer is only initialized with warm_up
        assert deserialized.weave_init_kwargs == {"autopatch_settings": AutopatchSettings(disable_autopatch=True)}

    def test_pipeline_tracing(self, mock_weave_client: Mock, sample_pipeline: Pipeline) -> None:
        """Test that pipeline operations are correctly traced"""

        connector = WeaveConnector(pipeline_name="test_pipeline")
        sample_pipeline.add_component("weave", connector)
        sample_pipeline.run(data={"greeting": "Hello"})

        # Verify WeaveClient interactions
        # Should create calls for pipeline and component operations
        mock_weave_client.create_call.assert_called()

        # Get all create_call arguments
        create_call_args: list[dict[str, Any]] = [call[1] for call in mock_weave_client.create_call.call_args_list]

        # Verify pipeline operation was traced
        pipeline_calls = [args for args in create_call_args if args.get("op") == "haystack.pipeline.run"]
        assert len(pipeline_calls) > 0

        # Verify component operations were traced
        component_calls = [args for args in create_call_args if args.get("op") == "haystack.component.run"]
        assert len(component_calls) >= 2  # Should have at least two component runs

        # Verify finish_call was called for all operations
        assert mock_weave_client.finish_call.call_count >= 3  # Pipeline + 2 components

    def test_error_handling(self, mock_weave_client: Mock) -> None:
        """Test that errors in pipeline execution are properly traced"""
        pp = Pipeline()
        pp.add_component("failing", FailingComponent())
        pp.add_component("weave", WeaveConnector(pipeline_name="test_pipeline"))

        # Run pipeline and expect exception
        with pytest.raises(ValueError):
            pp.run(data={})

        # Verify error was traced
        finish_call_args = mock_weave_client.finish_call.call_args_list
        error_traces = [args for args in finish_call_args if args[1].get("exception") is not None]
        assert len(error_traces) > 0

    def test_run_method(self) -> None:
        """Test the basic run method of WeaveConnector"""
        connector = WeaveConnector(pipeline_name="test_pipeline")
        result: dict[str, str] = connector.run()
        assert result == {"pipeline_name": "test_pipeline"}

    @pytest.mark.skipif(
        not os.environ.get("WANDB_API_KEY", None),
        reason="Export an env var called WANDB_API_KEY containing the Weights & Bias API key to run this test.",
    )
    @pytest.mark.integration
    def test_warmup_initializes_tracer(self, monkeypatch) -> None:
        """Test that warm_up initializes the tracer correctly"""
        monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
        connector = WeaveConnector(pipeline_name="test_pipeline")
        assert connector.tracer is None

        connector.warm_up()  # initialize tracer

        assert connector.tracer is not None
        assert isinstance(connector.tracer, WeaveTracer)
        assert haystack_configured_tracer.is_content_tracing_enabled is True

    @pytest.mark.skipif(
        not os.environ.get("WANDB_API_KEY", None),
        reason="Export an env var called WANDB_API_KEY containing the Weights & Bias API key to run this test.",
    )
    @pytest.mark.integration
    def test_warmup_initializes_tracer_with_weave_init_kwargs(self, monkeypatch) -> None:
        """Test that warm_up initializes the tracer correctly"""
        monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
        connector = WeaveConnector(
            pipeline_name="test_pipeline",
            weave_init_kwargs={"autopatch_settings": AutopatchSettings(disable_autopatch=True)},
        )
        assert connector.tracer is None

        connector.warm_up()  # initialize tracer

        assert connector.tracer is not None
        assert isinstance(connector.tracer, WeaveTracer)
        assert haystack_configured_tracer.is_content_tracing_enabled is True
