from haystack import component, tracing
from haystack_integrations.tracing.langfuse import LangfuseTracer

from langfuse import Langfuse


@component
class LangfuseConnector:
    """
    LangfuseConnector connects Haystack LLM framework with Langfuse in order to enable the tracing of operations
    and data flow within various components of a pipeline.

    Simply add this component to your pipeline, but *do not* connect it to any other component. The LangfuseConnector
    will automatically trace the operations and data flow within the pipeline.

    Note that you need to set the `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` environment variables in order
    to use this component. The `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are the secret and public keys provided
    by Langfuse. You can get these keys by signing up for an account on the Langfuse website.

    In addition, you need to set the `HAYSTACK_CONTENT_TRACING_ENABLED` environment variable to `true` in order to
    enable Haystack tracing in your pipeline.

    Here is an example of how to use it:

    ```python
    import os

    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    from haystack import Pipeline
    from haystack.components.builders import DynamicChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.connectors.langfuse import LangfuseConnector

    if __name__ == "__main__":

        pipe = Pipeline()
        pipe.add_component("tracer", LangfuseConnector("Chat example"))
        pipe.add_component("prompt_builder", DynamicChatPromptBuilder())
        pipe.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

        pipe.connect("prompt_builder.prompt", "llm.messages")

        messages = [
            ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
            ChatMessage.from_user("Tell me about {{location}}"),
        ]

        response = pipe.run(
            data={"prompt_builder": {"template_variables": {"location": "Berlin"}, "prompt_source": messages}}
        )
        print(response["llm"]["replies"][0])
        print(response["tracer"]["trace_url"])
    ```

    """

    def __init__(self, name: str, public: bool = False):
        """
        Initialize the LangfuseConnector component.

        :param name: The name of the pipeline or component. This name will be used to identify the tracing run on the
            Langfuse dashboard.
        :param public: Whether the tracing data should be public or private. If set to `True`, the tracing data will be
            publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private and
            only accessible to the Langfuse account owner. The default is `False`.
        """
        self.name = name
        self.tracer = LangfuseTracer(tracer=Langfuse(), name=name, public=public)
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str, trace_url=str)
    def run(self):
        """
        Runs the LangfuseConnector component.

        :returns: A dictionary with the following keys:
            - `name`: The name of the tracing component.
            - `trace_url`: The URL to the tracing data.
        """
        return {"name": self.name, "trace_url": self.tracer.get_trace_url()}
