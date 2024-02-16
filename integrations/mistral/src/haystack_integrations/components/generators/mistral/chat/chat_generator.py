# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, Optional

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret


@component
class MistralChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Mistral's large language models (LLMs).
    Currently supports `mistral-tiny`, `mistral-small` and `mistral-medium`
    models accessed through the chat completions API endpoint.

    Users can pass any text generation parameters valid for the `openai.ChatCompletion.create` method
    directly to this component via the `**generation_kwargs` parameter in __init__ or the `**generation_kwargs`
    parameter in `run` method.

    For more details on the parameters supported by the Mistral API, refer to the
    [Mistral API Docs](https://docs.mistral.ai/api/).

    ```python
    from haystack_integrations.components.generators.mistral import MistralChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = MistralChatGenerator()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
    >>meta={'model': 'mistral-tiny', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}

    ```

     Key Features and Compatibility:
         - **Primary Compatibility**: Designed to work seamlessly with the Mistral API Chat Completion endpoint.
         - **Streaming Support**: Supports streaming responses from the Mistral API Chat Completion endpoint.
         - **Customizability**: Supports all parameters supported by the Mistral API Chat Completion endpoint.

     Input and Output Format:
         - **ChatMessage Format**: This component uses the ChatMessage format for structuring both input and output,
           ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
           Details on the ChatMessage format can be found at: https://github.com/openai/openai-python/blob/main/chatml.md.
           Note that the Mistral API does not accept `system` messages yet. You can use `user` and `assistant` messages.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-tiny",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = "https://api.mistral.ai/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of MistralChatGenerator. Unless specified otherwise in the `model`, this is for Mistral's
        `mistral-tiny` model.

        :param api_key: The Mistral API key.
        :param model: The name of the Mistral chat completion model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url: The Mistral API Base url, defaults to `https://api.mistral.ai/v1`.
                             For more details, see Mistral [docs](https://docs.mistral.ai/api/).
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the Mistrak endpoint. See [Mistral API docs](https://docs.mistral.ai/api/t) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `safe_prompt`: Whether to inject a safety prompt before all conversations.
            - `random_seed`: The seed to use for random sampling.
        """
        super(MistralChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
        )
