# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

import httpx
from haystack import component, default_from_dict, default_to_dict, logging, tracing
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.base_serialization import deserialize_class_instance, serialize_class_instance
from langfuse import Langfuse

from haystack_integrations.tracing.langfuse import LangfuseTracer, SpanHandler

logger = logging.getLogger(__name__)


@component
class LangfuseConnector:
    """
    LangfuseConnector connects Haystack LLM framework with [Langfuse](https://langfuse.com) in order to enable the
    tracing of operations and data flow within various components of a pipeline.

    To use LangfuseConnector, add it to your pipeline without connecting it to any other components.
    It will automatically trace all pipeline operations when tracing is enabled.

    **Environment Configuration:**
    - `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY`: Required Langfuse API credentials.
    - `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be set to `"true"` to enable tracing.
    - `HAYSTACK_LANGFUSE_ENFORCE_FLUSH`: (Optional) If set to `"false"`, disables flushing after each component.
      Be cautious: this may cause data loss on crashes unless you manually flush before shutdown.
      By default, the data is flushed after each component and blocks the thread until the data is sent to Langfuse.

    If you disable flushing after each component make sure you will call langfuse.flush() explicitly before the
    program exits. For example:

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

    Here is an example of how to use LangfuseConnector in a pipeline:

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

    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector("Chat example"))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

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
    print(response["tracer"]["trace_id"])
    ```

    For advanced use cases, you can also customize how spans are created and processed by providing a custom
    SpanHandler. This allows you to add custom metrics, set warning levels, or attach additional metadata to your
    Langfuse traces:

    ```python
    from haystack_integrations.tracing.langfuse import DefaultSpanHandler, LangfuseSpan
    from typing import Optional

    class CustomSpanHandler(DefaultSpanHandler):

        def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
            # Custom span handling logic, customize Langfuse spans however it fits you
            # see DefaultSpanHandler for how we create and process spans by default
            pass

    connector = LangfuseConnector(span_handler=CustomSpanHandler())
    ```
    """

    def __init__(
        self,
        name: str,
        public: bool = False,
        public_key: Optional[Secret] = Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),  # noqa: B008
        secret_key: Optional[Secret] = Secret.from_env_var("LANGFUSE_SECRET_KEY"),  # noqa: B008
        httpx_client: Optional[httpx.Client] = None,
        span_handler: Optional[SpanHandler] = None,
        *,
        host: Optional[str] = None,
        langfuse_client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the LangfuseConnector component.

        :param name: The name for the trace. This name will be used to identify the tracing run in the Langfuse
            dashboard.
        :param public: Whether the tracing data should be public or private. If set to `True`, the tracing data will be
            publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private and
            only accessible to the Langfuse account owner. The default is `False`.
        :param public_key: The Langfuse public key. Defaults to reading from LANGFUSE_PUBLIC_KEY environment variable.
        :param secret_key: The Langfuse secret key. Defaults to reading from LANGFUSE_SECRET_KEY environment variable.
        :param httpx_client: Optional custom httpx.Client instance to use for Langfuse API calls. Note that when
            deserializing a pipeline from YAML, any custom client is discarded and Langfuse will create its own default
            client, since HTTPX clients cannot be serialized.
        :param span_handler: Optional custom handler for processing spans. If None, uses DefaultSpanHandler.
            The span handler controls how spans are created and processed, allowing customization of span types
            based on component types and additional processing after spans are yielded. See SpanHandler class for
            details on implementing custom handlers.
        host: Host of Langfuse API. Can also be set via `LANGFUSE_HOST` environment variable.
            By default it is set to `https://cloud.langfuse.com`.
        :param langfuse_client_kwargs: Optional custom configuration for the Langfuse client. This is a dictionary
            containing any additional configuration options for the Langfuse client. See the Langfuse documentation
            for more details on available configuration options.
        """
        self.name = name
        self.public = public
        self.secret_key = secret_key
        self.public_key = public_key
        self.span_handler = span_handler
        self.host = host
        self.langfuse_client_kwargs = langfuse_client_kwargs
        resolved_langfuse_client_kwargs = {
            "secret_key": secret_key.resolve_value() if secret_key else None,
            "public_key": public_key.resolve_value() if public_key else None,
            "httpx_client": httpx_client,
            "host": host,
            **(langfuse_client_kwargs or {}),
        }
        self.tracer = LangfuseTracer(
            tracer=Langfuse(**resolved_langfuse_client_kwargs),
            name=name,
            public=public,
            span_handler=span_handler,
        )
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str, trace_url=str, trace_id=str)
    def run(self, invocation_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Runs the LangfuseConnector component.

        :param invocation_context: A dictionary with additional context for the invocation. This parameter
            is useful when users want to mark this particular invocation with additional information, e.g.
            a run id from their own execution framework, user id, etc. These key-value pairs are then visible
            in the Langfuse traces.
        :returns: A dictionary with the following keys:
            - `name`: The name of the tracing component.
            - `trace_url`: The URL to the tracing data.
            - `trace_id`: The ID of the trace.
        """
        logger.debug(
            "Langfuse tracer invoked with the following context: '{invocation_context}'",
            invocation_context=invocation_context,
        )
        return {"name": self.name, "trace_url": self.tracer.get_trace_url(), "trace_id": self.tracer.get_trace_id()}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        span_handler = serialize_class_instance(self.span_handler) if self.span_handler else None
        if self.langfuse_client_kwargs:
            # pop httpx_client and mask from self._langfuse_client_kwargs to prevent serialization issues
            langfuse_client_kwargs = {
                k: v for k, v in self.langfuse_client_kwargs.items() if k not in ["httpx_client", "mask"]
            }
        else:
            langfuse_client_kwargs = None
        return default_to_dict(
            self,
            name=self.name,
            public=self.public,
            secret_key=self.secret_key.to_dict() if self.secret_key else None,
            public_key=self.public_key.to_dict() if self.public_key else None,
            # Note: httpx_client is not serialized as it's not serializable
            span_handler=span_handler,
            host=self.host,
            langfuse_client_kwargs=langfuse_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangfuseConnector":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["secret_key", "public_key"])
        init_params["span_handler"] = (
            deserialize_class_instance(init_params["span_handler"]) if init_params["span_handler"] else None
        )
        return default_from_dict(cls, data)
