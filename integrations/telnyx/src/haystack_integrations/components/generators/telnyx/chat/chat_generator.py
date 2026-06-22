# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel


@component
class TelnyxChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Telnyx Inference chat completion models.

    Telnyx exposes an OpenAI-compatible Chat Completions API at
    `https://api.telnyx.com/v2/ai/openai`. Users can pass any supported chat completion parameters directly
    to the component using the `generation_kwargs` parameter in `__init__` or in the `run` method.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.telnyx import TelnyxChatGenerator

    generator = TelnyxChatGenerator()
    result = generator.run([ChatMessage.from_user("Tell me about Telnyx Inference.")])
    print(result["replies"][0].text)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "moonshotai/Kimi-K2.6",
        "zai-org/GLM-5.1-FP8",
        "MiniMaxAI/MiniMax-M2.7",
    ]
    """A non-exhaustive list of model ids available through Telnyx Inference.
    Use the Telnyx OpenAI-compatible models endpoint for the current list available to your account."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TELNYX_API_KEY"),
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://api.telnyx.com/v2/ai/openai",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a TelnyxChatGenerator component.

        :param api_key:
            The Telnyx API key.
        :param model:
            The Telnyx Inference chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param api_base_url:
            The Telnyx OpenAI-compatible API base URL.
        :param generation_kwargs:
            Additional keyword arguments sent directly to the Telnyx chat completions endpoint.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls.
        :param timeout:
            Timeout for Telnyx client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Telnyx after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
        """
        super(TelnyxChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
            tools=tools,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        generation_kwargs = self.generation_kwargs.copy()
        response_format = generation_kwargs.get("response_format")
        if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            generation_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": to_strict_json_schema(response_format),
                },
            }

        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=generation_kwargs,
            tools=serialize_tools_or_toolset(self.tools),
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
