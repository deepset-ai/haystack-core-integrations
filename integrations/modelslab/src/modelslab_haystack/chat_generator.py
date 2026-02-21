# SPDX-FileCopyrightText: 2024-present ModelsLab
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType
from haystack.utils import Secret

MODELSLAB_API_BASE = "https://modelslab.com/uncensored-chat/v1"
MODELSLAB_DEFAULT_MODEL = "llama-3.1-8b-uncensored"


@component
class ModelsLabChatGenerator(OpenAIChatGenerator):
    """
    Completes chats using ModelsLab's uncensored Llama 3.1 models.

    `ModelsLabChatGenerator` connects to ModelsLab's OpenAI-compatible API
    to provide uncensored Llama 3.1 8B and 70B models with 128K context windows.
    It extends `OpenAIChatGenerator` with ModelsLab-specific defaults, including
    the correct ``api_base_url`` and ``MODELSLAB_API_KEY`` environment variable.

    Models:
        - ``llama-3.1-8b-uncensored`` — fast, efficient (default)
        - ``llama-3.1-70b-uncensored`` — higher quality, deeper reasoning

    Usage example:

    ```python
    from modelslab_haystack import ModelsLabChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = ModelsLabChatGenerator()   # reads MODELSLAB_API_KEY from env

    messages = [ChatMessage.from_user("Write a Python function to merge two sorted lists.")]
    response = generator.run(messages=messages)
    print(response["replies"][0].text)
    ```

    With streaming:

    ```python
    generator = ModelsLabChatGenerator(
        model="llama-3.1-70b-uncensored",
        streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),
    )
    generator.run(messages=[ChatMessage.from_user("Explain transformers.")])
    ```

    Get your API key at: https://modelslab.com
    API docs: https://docs.modelslab.com/uncensored-chat
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MODELSLAB_API_KEY"),
        model: str = MODELSLAB_DEFAULT_MODEL,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool = False,
        http_client_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize ``ModelsLabChatGenerator``.

        :param api_key: ModelsLab API key.
            Reads ``MODELSLAB_API_KEY`` from the environment by default.
            Get your key at https://modelslab.com.
        :param model: ModelsLab model to use.
            Defaults to ``llama-3.1-8b-uncensored`` (128K context).
            Also available: ``llama-3.1-70b-uncensored``.
        :param streaming_callback: Optional callback for streaming responses.
        :param generation_kwargs: Additional parameters for the API call
            (e.g., ``temperature``, ``max_tokens``, ``top_p``).
        :param timeout: Timeout in seconds for API requests.
        :param max_retries: Number of retries on transient failures.
        :param tools: Tools (functions) the model can call.
        :param tools_strict: Whether to enforce strict tool schemas.
        :param http_client_kwargs: Extra kwargs for the underlying httpx client.
        """
        super().__init__(
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=MODELSLAB_API_BASE,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            tools=tools,
            tools_strict=tools_strict,
            http_client_kwargs=http_client_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=self.streaming_callback,
            api_base_url=MODELSLAB_API_BASE,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelsLabChatGenerator":
        return default_from_dict(cls, data)
