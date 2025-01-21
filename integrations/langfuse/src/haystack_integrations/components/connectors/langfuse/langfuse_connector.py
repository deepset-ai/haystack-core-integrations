from typing import Any, Dict, Optional

import httpx
from haystack import component, default_from_dict, default_to_dict, logging, tracing
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.tracing.langfuse import LangfuseTracer
from langfuse import Langfuse

logger = logging.getLogger(__name__)


@component
class LangfuseConnector:
    """
    LangfuseConnector connects Haystack LLM framework with [Langfuse](https://langfuse.com) in order to enable the
    tracing of operations and data flow within various components of a pipeline.

    Simply add this component to your pipeline, but *do not* connect it to any other component. The LangfuseConnector
    will automatically trace the operations and data flow within the pipeline.

    Note that you need to set the `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` environment variables in order
    to use this component. The `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are the secret and public keys provided
    by Langfuse. You can get these keys by signing up for an account on the Langfuse website.

    In addition, you need to set the `HAYSTACK_CONTENT_TRACING_ENABLED` environment variable to `true` in order to
    enable Haystack tracing in your pipeline.

    Lastly, you may disable flushing the data after each component by setting the `HAYSTACK_LANGFUSE_ENFORCE_FLUSH`
    environent variable to `false`. By default, the data is flushed after each component and blocks the thread until
    the data is sent to Langfuse. **Caution**: Disabling this feature may result in data loss if the program crashes
    before the data is sent to Langfuse. Make sure you will call langfuse.flush() explicitly before the program exits.
    E.g. by using tracer.actual_tracer.flush():

    ```python
    from haystack.tracing import tracer

    try:
        # your code here
    finally:
        tracer.actual_tracer.flush()
    ```
    or in FastAPI by defining a shutdown event handler:
    ```python
    from haystack.tracing import tracer

    # ...


    @app.on_event("shutdown")
    async def shutdown_event():
        tracer.actual_tracer.flush()
    ```

    Here is an example of how to use it:

    ```python
    import os

    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    from haystack import Pipeline
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.connectors.langfuse import (
        LangfuseConnector,
    )

    if __name__ == "__main__":
        pipe = Pipeline()
        pipe.add_component("tracer", LangfuseConnector("Chat example"))
        pipe.add_component("prompt_builder", ChatPromptBuilder())
        pipe.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

        pipe.connect("prompt_builder.prompt", "llm.messages")

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
        print(response["tracer"]["trace_url"])
    ```

    """

    def __init__(
        self,
        name: str,
        public: bool = False,
        public_key: Optional[Secret] = Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),  # noqa: B008
        secret_key: Optional[Secret] = Secret.from_env_var("LANGFUSE_SECRET_KEY"),  # noqa: B008
        httpx_client: Optional[httpx.Client] = None,
    ):
        """
        Initialize the LangfuseConnector component.

        :param name: The name of the pipeline or component. This name will be used to identify the tracing run on the
            Langfuse dashboard.
        :param public: Whether the tracing data should be public or private. If set to `True`, the tracing data will be
            publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private and
            only accessible to the Langfuse account owner. The default is `False`.
        :param public_key: The Langfuse public key. Defaults to reading from LANGFUSE_PUBLIC_KEY environment variable.
        :param secret_key: The Langfuse secret key. Defaults to reading from LANGFUSE_SECRET_KEY environment variable.
        :param httpx_client: Optional custom httpx.Client instance to use for Langfuse API calls. Note that when
            deserializing a pipeline from YAML, any custom client is discarded and Langfuse will create its own default
            client, since HTTPX clients cannot be serialized.
        """
        self.name = name
        self.public = public
        self.secret_key = secret_key
        self.public_key = public_key
        self.tracer = LangfuseTracer(
            tracer=Langfuse(
                secret_key=secret_key.resolve_value() if secret_key else None,
                public_key=public_key.resolve_value() if public_key else None,
                httpx_client=httpx_client,
            ),
            name=name,
            public=public,
        )
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str, trace_url=str)
    def run(self, invocation_context: Optional[Dict[str, Any]] = None):
        """
        Runs the LangfuseConnector component.

        :param invocation_context: A dictionary with additional context for the invocation. This parameter
            is useful when users want to mark this particular invocation with additional information, e.g.
            a run id from their own execution framework, user id, etc. These key-value pairs are then visible
            in the Langfuse traces.
        :returns: A dictionary with the following keys:
            - `name`: The name of the tracing component.
            - `trace_url`: The URL to the tracing data.
        """
        logger.debug(
            "Langfuse tracer invoked with the following context: '{invocation_context}'",
            invocation_context=invocation_context,
        )
        return {"name": self.name, "trace_url": self.tracer.get_trace_url()}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            name=self.name,
            public=self.public,
            secret_key=self.secret_key.to_dict() if self.secret_key else None,
            public_key=self.public_key.to_dict() if self.public_key else None,
            # Note: httpx_client is not serialized as it's not serializable
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangfuseConnector":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["secret_key", "public_key"])
        return default_from_dict(cls, data)
