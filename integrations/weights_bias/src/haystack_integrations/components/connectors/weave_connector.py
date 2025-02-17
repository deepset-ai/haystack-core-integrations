from typing import Any, Dict, Optional

from haystack import component, default_from_dict, default_to_dict, tracing
from haystack.tracing import tracer as haystack_tracer

from haystack_integrations.tracing.weave import WeaveTracer


@component
class WeaveConnector:
    """Collects traces from your pipeline and sends them to Weights & Biases.

    Add this component to your pipeline to integrate with the Weights & Biases Weave framework and
    use their LLM evaluation features.
    """

    def __init__(self, pipeline_name: str) -> None:
        """
        Initialize WeaveConnector.

        :param pipeline_name: The name of the pipeline you want to trace.

        """
        self.pipeline_name = pipeline_name

        # TODO: This is a hack because content tracing enabled seems to rely on import order which is hard to debug.
        # As a workaround, we assume that users adding the WeaveConnector to a pipeline always want to trace content.
        haystack_tracer.is_content_tracing_enabled = True

        # We defer initializing the tracer to component warmup.
        self.tracer: Optional[WeaveTracer] = None

    def warm_up(self) -> None:
        if self.tracer is None:
            self.tracer = WeaveTracer(project_name=self.pipeline_name)
            tracing.enable_tracing(self.tracer)

    @component.output_types(no_op=str)
    def run(self, no_op: str = "no_op") -> Dict[str, str]:
        return {"no_op": no_op}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, pipeline_name=self.pipeline_name)  # type: ignore

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaveConnector":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)  # type: ignore
