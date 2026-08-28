# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.tools import ToolsType, flatten_tools_or_toolsets
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from anthropic import Anthropic
from anthropic.types import ToolParam
from haystack_integrations.components.generators.anthropic.chat.utils import convert_messages_to_anthropic_format


class AnthropicTokenCounter:
    """
    Counts input tokens for Anthropic models using the Anthropic token counting API.

    Uses the `POST /v1/messages/count_tokens` endpoint, which returns an exact token
    count without generating a response or incurring generation costs.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.token_counters.anthropic import AnthropicTokenCounter

    counter = AnthropicTokenCounter(model="claude-sonnet-4-5")
    messages = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("How many tokens is this?"),
    ]
    token_count = counter.count(messages)
    print(token_count)
    ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Secret = Secret.from_env_var("ANTHROPIC_API_KEY"),  # noqa: B008
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Create an AnthropicTokenCounter.

        :param model: The Anthropic model to use for tokenization. Token counts are
            model-specific; always count against the model you intend to use.
        :param api_key: The Anthropic API key. Defaults to the `ANTHROPIC_API_KEY`
            environment variable.
        :param timeout: HTTP timeout in seconds for the Anthropic client.
        :param max_retries: Maximum number of retries for failed requests.
        """
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        client_kwargs: dict[str, Any] = {"api_key": api_key.resolve_value()}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self._client = Anthropic(**client_kwargs)

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Count the tokens for the given messages and optional tools.

        :param messages: The list of ChatMessages to count tokens for.
        :param tools: Optional list of Tools whose schemas are included in the count.
        :returns: The number of input tokens.
        """
        if not messages:
            return 0

        system_messages, non_system_messages = convert_messages_to_anthropic_format(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": non_system_messages,
        }
        if system_messages:
            kwargs["system"] = system_messages
        if tools:
            flattened = flatten_tools_or_toolsets(tools)
            kwargs["tools"] = [
                ToolParam(name=t.name, description=t.description or "", input_schema=t.parameters) for t in flattened
            ]

        response = self._client.messages.count_tokens(**kwargs)
        return response.input_tokens

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this token counter to a dictionary.

        :returns: The serialized token counter.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnthropicTokenCounter":
        """
        Deserialize a token counter from a dictionary.

        :param data: The dictionary to deserialize from.
        :returns: The deserialized token counter.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
