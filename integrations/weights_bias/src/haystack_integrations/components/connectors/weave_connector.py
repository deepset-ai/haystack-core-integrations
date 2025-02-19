import os
from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict, logging, tracing
from haystack.tracing import tracer as haystack_tracer

from haystack_integrations.tracing.weave import WeaveTracer

logger = logging.getLogger(__name__)


@component
class WeaveConnector:
    """
    Collects traces from your pipeline and sends them to Weights & Biases.

    Add this component to your pipeline to integrate with the Weights & Biases Weave framework for tracing and
    monitoring your pipeline components.

    Note that you need to have the `WANDB_API_KEY` environment variable set to your Weights & Biases API key.

    NOTE: If you don't have a Weights & Biases account it will interactively ask you to set one and your input
    will then be stored in ~/.netrc

    In addition, you need to set the `HAYSTACK_CONTENT_TRACING_ENABLED` environment variable to `true` in order to
    enable Haystack tracing in your pipeline.
    """

    def __init__(self, pipeline_name: str) -> None:
        """
        Initialize WeaveConnector.

        :param pipeline_name: The name of the pipeline you want to trace.
        """
        self.pipeline_name = pipeline_name
        content_tracing_enabled = os.getenv("HAYSTACK_CONTENT_TRACING_ENABLED", "false").lower()
        self.enable_tracing = content_tracing_enabled == "true"
        haystack_tracer.is_content_tracing_enabled = self.enable_tracing
        if not haystack_tracer.is_content_tracing_enabled:
            logger.warning(
                "Traces will not be logged to Weave because Haystack tracing is disabled. "
                "To enable, set the HAYSTACK_CONTENT_TRACING_ENABLED environment variable to true "
                "before importing Haystack."
            )

        self.tracer: Optional[WeaveTracer] = None

    def warm_up(self) -> None:
        """Initialize the WeaveTracer."""
        if self.tracer is None:
            self.tracer = WeaveTracer(project_name=self.pipeline_name)
            tracing.enable_tracing(self.tracer)

    @component.output_types(no_op=str)
    def run(self, no_op: str = "no_op") -> dict[str, str]:
        return {"no_op": no_op}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with all the necessary information to recreate this component.
        """
        return default_to_dict(self, pipeline_name=self.pipeline_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaveConnector":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
