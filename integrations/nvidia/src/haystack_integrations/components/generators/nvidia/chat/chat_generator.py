# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, Optional

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

from haystack_integrations.utils.nvidia import DEFAULT_API_URL

logger = logging.getLogger(__name__)


@component
class NvidiaChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using NVIDIA generative models.
    For supported models, see [NVIDIA Docs](https://build.nvidia.com/models).

    Users can pass any text generation parameters valid for the NVIDIA Chat Completion API
    directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)

    For more details on the parameters supported by the NVIDIA API, refer to the
    [NVIDIA Docs](https://build.nvidia.com/models).

    Usage example:
    ```python
    from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = NvidiaChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("NVIDIA_API_KEY"),
        model: str = "meta/llama-3.1-8b-instruct",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of NvidiaChatGenerator.

        :param api_key:
            The NVIDIA API key.
        :param model:
            The name of the NVIDIA chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The NVIDIA API Base url.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the NVIDIA API endpoint. See [NVIDIA API docs](https://docs.nvcf.nvidia.com/ai/generative-models/)
            for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `response_format`: For NVIDIA NIM servers, this parameter has limited support.
                - The basic JSON mode with `{"type": "json_object"}` is supported by compatible models, to produce
                valid JSON output.
                To pass the JSON schema to the model, use the `guided_json` parameter in `extra_body`.
                For example:
                ```python
                generation_kwargs={
                    "extra_body": {
                        "nvext": {
                            "guided_json": {
                                json_schema
                        }
                    }
                }
                ```
                For more details, see the [NVIDIA NIM documentation](https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html).
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance.
        :param timeout:
            The timeout for the NVIDIA API call.
        :param max_retries:
            Maximum number of retries to contact NVIDIA after an internal error.
            If not set, it defaults to either the `NVIDIA_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(NvidiaChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            tools=tools,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=serialize_tools_or_toolset(self.tools),
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
