# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_from_dict
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, deserialize_tools_or_toolset_inplace
from haystack.utils import deserialize_callable
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.vllm import VLLMChatGenerator


@component
class STACKITChatGenerator(VLLMChatGenerator):
    """
    Enables text generation using STACKIT generative models through their model serving service.

    Users can pass any text generation parameters valid for the STACKIT Chat Completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    ### Usage example
    ```python
    from haystack_integrations.components.generators.stackit import STACKITChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = STACKITChatGenerator(model="cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic")

    result = generator.run([ChatMessage.from_user("Tell me a joke.")])
    print(result)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "Qwen/Qwen3.6-27B",
        "cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic",
        "openai/gpt-oss-120b",
        "google/gemma-3-27b-it",
        "openai/gpt-oss-20b",
    ]
    """A non-exhaustive list of chat models supported by this component.
    See https://docs.stackit.cloud/products/data-and-ai/ai-model-serving/basics/available-shared-models
    for the full list."""

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        generation_kwargs: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
    ) -> None:
        """
        Creates an instance of STACKITChatGenerator class.

        :param model:
            The name of the chat completion model to use.
        :param api_key:
            The STACKIT API key.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The STACKIT API Base url.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the STACKIT endpoint.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `safe_prompt`: Whether to inject a safety prompt before all conversations.
            - `random_seed`: The seed to use for random sampling.
            - `response_format`: A response format dictionary containing a JSON schema.
                For Pydantic models, use `model_json_schema()` to build the schema;
                passing the model class directly is not supported.
        :param timeout:
            Timeout for STACKIT client calls. If not set, uses the OpenAI SDK default.
        :param max_retries:
            Maximum number of retries to contact STACKIT after an internal error.
            If not set, uses the OpenAI SDK default.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name. Not all models support tools.
        """
        super(STACKITChatGenerator, self).__init__(  # noqa: UP008
            model=model,
            api_key=api_key,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
            tools=tools,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "STACKITChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        init_params = data["init_parameters"]
        # pop the tool_strict legacy field
        init_params.pop("tools_strict", None)
        deserialize_tools_or_toolset_inplace(init_params, key="tools")
        if callback := init_params.get("streaming_callback"):
            init_params["streaming_callback"] = deserialize_callable(callback)
        return default_from_dict(cls, data)
