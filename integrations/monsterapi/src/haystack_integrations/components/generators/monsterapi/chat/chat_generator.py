# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, Optional

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret


@component
class MonsterChatGeneraor(OpenAIChatGenerator):
    """
    Enables text generation using Monster AI generative models.
    For supported models, see [Monster API docs](https://developer.monsterapi.ai/).

    Users can pass any text generation parameters valid for the Monster Chat Completion API
    directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the MonsterAPI Chat Completion endpoint.
    - **Streaming Support**: Supports streaming responses from the MonsterAPI Chat Completion endpoint.
    - **Customizability**: Supports all parameters supported by the MonsterAPI Chat Completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage)

    For more details on the parameters supported by the Monster API, refer to the
    [MonsterAPI Docs](https://developer.monsterapi.ai/).

    Usage example:
    ```python
    from haystack_integrations.components.generators.monsterapi import MonsterChatGeneraor
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = MonsterChatGeneraor()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
    >>meta={'model': 'meta-llama/Meta-Llama-3-8B-Instruct', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MONSTER_API_KEY"),
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = "https://llm.monsterapi.ai/v1/",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of MonsterChatGeneraor. Unless specified otherwise in the `model`, this is for Monster's
        `meta-llama/Meta-Llama-3-8B-Instruct` model.

        :param api_key:
            The Monster API key.
        :param model:
            The name of the Monster chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Monster API Base url.
            For more details, see MonsterAPI [docs](https://developer.monsterapi.ai/).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Monster endpoint. See [MonsterAPI docs](https://developer.monsterapi.ai/) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `random_seed`: The seed to use for random sampling.
        """
        super(MonsterChatGeneraor, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
        )
