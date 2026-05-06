# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from typing import Any, ClassVar

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import (
    Secret,
    deserialize_callable,
    deserialize_secrets_inplace,
    serialize_callable,
)

from anthropic import AnthropicFoundry, AsyncAnthropicFoundry

from .chat_generator import AnthropicChatGenerator

logger = logging.getLogger(__name__)


@component
class AnthropicFoundryChatGenerator(AnthropicChatGenerator):
    """
    Enables text generation using Anthropic's Claude models via Azure Foundry.

    A variety of Claude models (Opus, Sonnet, Haiku, and others) are available through Azure Foundry.

    To use AnthropicFoundryChatGenerator, you must have an Azure subscription with Foundry enabled
    and the desired Anthropic model deployed in your Foundry resource.

    For more details, refer to the [Anthropic Foundry documentation](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/foundry.md).

    Any valid text generation parameters for the Anthropic messaging API can be passed to
    the AnthropicFoundry API. Users can provide these parameters directly to the component via
    the `generation_kwargs` parameter in `__init__` or the `run` method.

    For more details on the parameters supported by the Anthropic API, refer to the
    Anthropic Message API [documentation](https://docs.anthropic.com/en/api/messages).

    ```python
    from haystack_integrations.components.generators.anthropic import AnthropicFoundryChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = AnthropicFoundryChatGenerator(
        model="claude-sonnet-4-5",
        api_key=Secret.from_env_var("ANTHROPIC_FOUNDRY_API_KEY"),
        resource="my-resource",
    )

    response = client.run(messages)
    print(response)
    >> {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
    >> "Natural Language Processing (NLP) is a field of artificial intelligence that
    >> focuses on enabling computers to understand, interpret, and generate human language. It involves developing
    >> techniques and algorithms to analyze and process text or speech data, allowing machines to comprehend and
    >> communicate in natural languages like English, Spanish, or Chinese.")],
    >> _name=None, _meta={'model': 'claude-sonnet-4-5', 'index': 0, 'finish_reason': 'end_turn',
    >> 'usage': {'input_tokens': 15, 'output_tokens': 64}})]}
    ```

    For more details on supported models and their capabilities, refer to the Anthropic
    [documentation](https://docs.anthropic.com/claude/docs/intro-to-claude).
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-haiku-4-5",
    ]
    """A non-exhaustive list of chat models supported by this component.
    The actual availability depends on your Azure Foundry resource configuration."""

    def __init__(
        self,
        *,
        api_key: Secret | None = Secret.from_env_var("ANTHROPIC_FOUNDRY_API_KEY", strict=True),  # noqa: B008
        resource: str | None = None,
        endpoint: str | None = None,
        model: str = "claude-sonnet-4-5",
        streaming_callback: Callable[[StreamingChunk], None] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        ignore_tools_thinking_messages: bool = True,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
    ) -> None:
        """
        Creates an instance of AnthropicFoundryChatGenerator.

        :param api_key: The API key to use for authentication.
            Defaults to the `ANTHROPIC_FOUNDRY_API_KEY` environment variable.
            Can be `None` when using `azure_ad_token_provider` instead.
        :param resource: The Foundry resource name. Can also be set via the `ANTHROPIC_FOUNDRY_RESOURCE`
            environment variable. Either `resource` or `endpoint` must be provided.
        :param endpoint: The full Foundry endpoint URL (e.g.,
            "https://your-resource.openai.azure.com/anthropic").
            Either `resource` or `endpoint` must be provided.
        :param model: The name of the model to use (deployment name in Foundry).
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the AnthropicFoundry endpoint. See Anthropic [documentation](https://docs.anthropic.com/claude/reference/messages_post)
            for more details.
            Supported generation_kwargs parameters are:
            - `system`: The system message to be passed to the model.
            - `max_tokens`: The maximum number of tokens to generate.
            - `metadata`: A dictionary of metadata to be passed to the model.
            - `stop_sequences`: A list of strings that the model should stop generating at.
            - `temperature`: The temperature to use for sampling.
            - `top_p`: The top_p value to use for nucleus sampling.
            - `top_k`: The top_k value to use for top-k sampling.
            - `extra_headers`: A dictionary of extra headers to be passed to the model (i.e. for beta features).
        :param ignore_tools_thinking_messages: Anthropic's approach to tools (function calling) resolution involves a
            "chain of thought" messages before returning the actual function names and parameters in a message. If
            `ignore_tools_thinking_messages` is `True`, the generator will drop so-called thinking messages when tool
            use is detected. See the Anthropic [tools](https://docs.anthropic.com/en/docs/tool-use#chain-of-thought-tool-use)
            for more details.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset, that the model can use.
            Each tool should have a unique name.
        :param timeout:
            Timeout for Anthropic client calls. If not set, it defaults to the default set by the Anthropic client.
        :param max_retries:
            Maximum number of retries to attempt for failed requests. If not set, it defaults to the default set by
            the Anthropic client.
        :param azure_ad_token_provider: A function that returns an Azure AD token for authentication.
            Can be used instead of `api_key` for enhanced security.
            See [Azure Identity documentation](https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/overview)
            for more details.
        """
        if api_key is None and azure_ad_token_provider is None:
            msg = "Please provide an API key or an azure_ad_token_provider."
            raise ValueError(msg)

        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        self.api_key: Secret | None = api_key  # type: ignore[assignment]
        self.resource = resource or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE")
        self.endpoint = endpoint
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.ignore_tools_thinking_messages = ignore_tools_thinking_messages
        self.tools = tools
        self.timeout = timeout
        self.max_retries = max_retries
        self.azure_ad_token_provider = azure_ad_token_provider

        if not self.resource and not self.endpoint:
            msg = (
                "Either 'resource' or 'endpoint' must be provided. "
                "Set ANTHROPIC_FOUNDRY_RESOURCE environment variable or pass resource/endpoint parameter."
            )
            raise ValueError(msg)

        # Clients are created lazily in warm_up()
        self.client = None  # type: ignore[assignment]
        self.async_client = None  # type: ignore[assignment]
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Create the AnthropicFoundry clients.

        This method is idempotent — it only creates clients once.
        """
        if self._is_warmed_up:
            return

        client_kwargs: dict[str, Any] = {}

        if self.azure_ad_token_provider:
            client_kwargs["azure_ad_token_provider"] = self.azure_ad_token_provider
        else:
            client_kwargs["api_key"] = self.api_key.resolve_value()  # type: ignore[union-attr]

        if self.endpoint:
            client_kwargs["base_url"] = self.endpoint
        else:
            client_kwargs["resource"] = self.resource

        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout

        if self.max_retries is not None:
            client_kwargs["max_retries"] = self.max_retries

        self.client = AnthropicFoundry(**client_kwargs)  # type: ignore[assignment]
        self.async_client = AsyncAnthropicFoundry(**client_kwargs)  # type: ignore[assignment]
        self._is_warmed_up = True

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Invokes the AnthropicFoundry API with the given messages and generation kwargs.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Optional arguments to pass to the Anthropic generation endpoint.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset, that the model can use.
            Each tool should have a unique name. If set, it will override the `tools` parameter set during component
            initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        if not self._is_warmed_up:
            self.warm_up()
        return super(AnthropicFoundryChatGenerator, self).run(  # noqa: UP008
            messages=messages, streaming_callback=streaming_callback, generation_kwargs=generation_kwargs, tools=tools
        )

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Async version of the run method. Invokes the AnthropicFoundry API with the given messages and generation kwargs.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Optional arguments to pass to the Anthropic generation endpoint.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset, that the model can use.
            Each tool should have a unique name. If set, it will override the `tools` parameter set during component
            initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        if not self._is_warmed_up:
            self.warm_up()
        return await super(AnthropicFoundryChatGenerator, self).run_async(  # noqa: UP008
            messages=messages, streaming_callback=streaming_callback, generation_kwargs=generation_kwargs, tools=tools
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        azure_ad_token_provider_name = (
            serialize_callable(self.azure_ad_token_provider) if self.azure_ad_token_provider else None
        )
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict() if self.api_key else None,
            resource=self.resource,
            endpoint=self.endpoint,
            model=self.model,
            streaming_callback=callback_name,
            generation_kwargs=self.generation_kwargs,
            ignore_tools_thinking_messages=self.ignore_tools_thinking_messages,
            tools=serialize_tools_or_toolset(self.tools),
            timeout=self.timeout,
            max_retries=self.max_retries,
            azure_ad_token_provider=azure_ad_token_provider_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnthropicFoundryChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

        init_params = data.get("init_parameters", {})
        if serialized_callback := init_params.get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback)
        if serialized_token_provider := init_params.get("azure_ad_token_provider"):
            data["init_parameters"]["azure_ad_token_provider"] = deserialize_callable(serialized_token_provider)

        return default_from_dict(cls, data)
