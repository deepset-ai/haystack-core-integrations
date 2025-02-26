from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict, logging, tracing

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

    To use this connector simply add it to your pipeline without any connections, and it will automatically start
    sending traces to Weights & Biases.

    Example:
    ```python
    import os

    from haystack import Pipeline
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    from haystack_integrations.components.connectors import WeaveConnector

    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    pipe = Pipeline()
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))
    pipe.connect("prompt_builder.prompt", "llm.messages")

    connector = WeaveConnector(pipeline_name="test_pipeline")
    pipe.add_component("weave", connector)

    messages = [
        ChatMessage.from_system(
            "Always respond in German even if some input data is in other languages."
        ),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(
        data={
            "prompt_builder": {
                "template_variables": {"location": "Berlin"},
                "template": messages,
            }
        }
    )
    print(response["llm"]["replies"][0])
    ```

    You should then head to `https://wandb.ai/<user_name>/projects` and see the complete trace for your pipeline under
    the pipeline name you specified, when creating the `WeaveConnector`

    """

    def __init__(self, pipeline_name: str) -> None:
        """
        Initialize WeaveConnector.

        :param pipeline_name: The name of the pipeline you want to trace.
        """
        self.pipeline_name = pipeline_name
        self.tracer: Optional[WeaveTracer] = None

    def warm_up(self) -> None:
        """Initialize the WeaveTracer."""
        if self.tracer is None:
            self.tracer = WeaveTracer(project_name=self.pipeline_name)
            tracing.enable_tracing(self.tracer)

    @component.output_types(pipeline_name=str)
    def run(self) -> dict[str, str]:
        # NOTE: this is a no-op component - it simply triggers the Tracer to sends traces to Weights & Biases
        return {"pipeline_name": self.pipeline_name}

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
