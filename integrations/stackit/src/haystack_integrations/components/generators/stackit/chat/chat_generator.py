# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, Optional, List

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import StreamingCallbackT
from haystack.dataclasses import StreamingChunk, ChatMessage, Tool
from haystack.utils.auth import Secret


@component
class STACKITChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using STACKIT generative models via their model serving service.

    Users can pass any text generation parameters valid for the STACKIT Chat Completion API
    directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage)

    Usage example:
    ```python
    from haystack_integrations.components.generators.stackit import STACKITChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8")

    result = generator.run([ChatMessage.from_user("Tell me a joke.")])
    print(result)
    ```
    """

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of class STACKITChatGenerator.

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
        """
        super(STACKITChatGenerator, self).__init__(  # noqa: UP008
            model=model,
            api_key=api_key,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
        )

    def _prepare_api_call(  # noqa: PLR0913
        self,
        *,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        prepared_api_call = super(STACKITChatGenerator, self)._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        prepared_api_call.pop("tools")
        return prepared_api_call