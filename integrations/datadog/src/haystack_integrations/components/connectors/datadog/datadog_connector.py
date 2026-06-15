# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ddtrace.trace import tracer as dd_tracer
from haystack import component, default_from_dict, default_to_dict, logging, tracing

from haystack_integrations.tracing.datadog import DatadogTracer

logger = logging.getLogger(__name__)


@component
class DatadogConnector:
    """
    DatadogConnector connects Haystack to [Datadog](https://www.datadoghq.com/) in order to enable the tracing of

    operations and data flow within the components of a pipeline.

    To use the DatadogConnector, add it to your pipeline without connecting it to any other component. It will
    automatically trace all pipeline operations when tracing is enabled.

    **Environment Configuration:**
    - `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be set to `"true"` to trace the content (inputs and outputs) of the
      pipeline components.
    - Datadog is configured through the standard `ddtrace` mechanisms, e.g. the `DD_SERVICE`, `DD_ENV` and
      `DD_VERSION` environment variables or by running your application with the `ddtrace-run` command. See the
      [ddtrace documentation](https://ddtrace.readthedocs.io/en/stable/) for more details.

    Here is an example of how to use the DatadogConnector in a pipeline:

    ```python
    import os

    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    from haystack import Pipeline
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.connectors.datadog import DatadogConnector

    pipe = Pipeline()
    pipe.add_component("tracer", DatadogConnector("Chat example"))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(
        data={"prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages}}
    )
    print(response["llm"]["replies"][0])
    ```
    """

    def __init__(self, name: str = "datadog") -> None:
        """
        Initialize the DatadogConnector component.

        :param name: The name used to identify this tracing component. It is returned by the `run` method and can be
            used to mark traces produced by this connector.
        """
        self.name = name
        self.tracer = DatadogTracer(tracer=dd_tracer)
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str)
    def run(self) -> dict[str, str]:
        """
        Runs the DatadogConnector component.

        :returns: A dictionary with the following keys:
            - `name`: The name of the tracing component.
        """
        logger.debug("Datadog tracer invoked.")
        return {"name": self.name}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(self, name=self.name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatadogConnector":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)
