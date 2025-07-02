# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, logging
from haystack.dataclasses import ChatMessage, StreamingCallbackT, select_streaming_callback

from .chat.chat_generator import WatsonxChatGenerator

logger = logging.getLogger(__name__)


@component
class WatsonxGenerator(WatsonxChatGenerator):
    """
    Enables text completions using IBM's watsonx.ai foundation models.

    This component extends WatsonxChatGenerator to provide the standard Generator interface
    that works with prompt strings instead of ChatMessage objects. It inherits all the
    functionality from WatsonxChatGenerator while adapting the input/output format.

    The generator works with IBM's foundation models including:
    - granite-13b-chat-v2
    - llama-2-70b-chat
    - llama-3-70b-instruct
    - Other watsonx.ai chat models

    You can customize the generation behavior by passing parameters to the
    watsonx.ai API through the `generation_kwargs` argument. These parameters
    are passed directly to the watsonx.ai inference endpoint.

    For details on watsonx.ai API parameters, see
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.watsonx.generator import (
        WatsonxGenerator,
    )
    from haystack.utils import Secret

    generator = WatsonxGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        model="ibm/granite-13b-chat-v2",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )

    response = generator.run(
        prompt="Explain quantum computing in simple terms",
        system_prompt="You are a helpful physics teacher.",
    )
    print(response)
    ```
    Output:
    ```
    {
        "replies": ["Quantum computing uses quantum-mechanical phenomena like...."],
        "meta": [
            {
                "model": "ibm/granite-13b-chat-v2",
                "project_id": "your-project-id",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 45,
                    "total_tokens": 57,
                },
            }
        ],
    }
    ```
    """

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(  # type: ignore[override]
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate text completions synchronously.

        :param prompt:
            The input prompt string for text generation.
        :param system_prompt:
            An optional system prompt to provide context or instructions for the generation.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided, this will override the `streaming_callback` set in the `__init__` method.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.
        :returns:
            A dictionary with the following keys:
            - `replies`: A list of generated text completions as strings.
            - `meta`: A list of metadata dictionaries containing information about each generation,
            including model name, finish reason, and token usage statistics.
        """
        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        messages = self._prepare_messages(prompt, system_prompt)

        chat_response = WatsonxChatGenerator.run(
            self, messages=messages, generation_kwargs=generation_kwargs, streaming_callback=streaming_callback
        )

        return self._convert_chat_response_to_generator_format(chat_response)

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    async def run_async(  # type: ignore[override]
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate text completions asynchronously.

        :param prompt:
            The input prompt string for text generation.
        :param system_prompt:
            An optional system prompt to provide context or instructions for the generation.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided, this will override the `streaming_callback` set in the `__init__` method.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.
        :returns:
            A dictionary with the following keys:
            - `replies`: A list of generated text completions as strings.
            - `meta`: A list of metadata dictionaries containing information about each generation,
            including model name, finish reason, and token usage statistics.
        """
        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )
        messages = self._prepare_messages(prompt, system_prompt)

        chat_response = await WatsonxChatGenerator.run_async(
            self, messages=messages, generation_kwargs=generation_kwargs, streaming_callback=streaming_callback
        )

        return self._convert_chat_response_to_generator_format(chat_response)

    def _prepare_messages(self, prompt: str, system_prompt: str | None = None) -> list[ChatMessage]:
        """
        Convert prompt and system_prompt to ChatMessage format.

        :param prompt: The user prompt
        :param system_prompt: Optional system prompt
        :return: List of ChatMessage objects
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage.from_system(system_prompt))

        messages.append(ChatMessage.from_user(prompt))

        return messages

    def _convert_chat_response_to_generator_format(self, chat_response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert ChatGenerator response format to Generator format.

        :param chat_response: Response from WatsonxChatGenerator
        :return: Response in Generator format with replies and meta lists
        """
        replies = []
        meta = []

        for chat_message in chat_response.get("replies", []):
            text_content = chat_message.text if hasattr(chat_message, "text") else str(chat_message)
            replies.append(text_content)

            message_meta = getattr(chat_message, "meta", {}) or {}
            meta.append(message_meta)

        return {"replies": replies, "meta": meta}
