# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging, tracing

import opentelemetry.trace
from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer

logger = logging.getLogger(__name__)


@component
class OpenTelemetryConnector:
    """
    OpenTelemetryConnector connects Haystack to [OpenTelemetry](https://opentelemetry.io/) in order to enable the

    tracing of operations and data flow within the components of a pipeline.

    To use the OpenTelemetryConnector, add it to your pipeline without connecting it to any other component. It will
    automatically trace all pipeline operations when tracing is enabled. Make sure to configure an OpenTelemetry
    `TracerProvider` (for example, with an exporter) before initializing the connector.

    **Environment Configuration:**
    - `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be set to `"true"` to trace the content (inputs and outputs) of the
      pipeline components.

    Here is an example of how to use the OpenTelemetryConnector in a pipeline:

    ```python
    import os

    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes

    # Configure the OpenTelemetry SDK. A service name is required for most backends.
    resource = Resource(attributes={ResourceAttributes.SERVICE_NAME: "haystack"})
    tracer_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    from haystack import Pipeline
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.connectors.opentelemetry import OpenTelemetryConnector

    pipe = Pipeline()
    pipe.add_component("tracer", OpenTelemetryConnector())
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

    def __init__(self, name: str = "opentelemetry") -> None:
        """
        Initialize the OpenTelemetryConnector component.

        :param name: The name used to identify this tracing component. It is returned by the `run` method and can be
            used to mark traces produced by this connector.
        """
        self.name = name
        self.tracer = OpenTelemetryTracer(opentelemetry.trace.get_tracer("haystack"))
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str)
    def run(self) -> dict[str, str]:
        """
        Runs the OpenTelemetryConnector component.

        :returns: A dictionary with the following keys:
            - `name`: The name of the tracing component.
        """
        logger.debug("OpenTelemetry tracer invoked.")
        return {"name": self.name}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(self, name=self.name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenTelemetryConnector":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)
