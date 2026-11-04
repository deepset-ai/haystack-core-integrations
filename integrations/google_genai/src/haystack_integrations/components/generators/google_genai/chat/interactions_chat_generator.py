# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, ClassVar, Literal

from google.genai import Client
from google.genai.client import AsyncClient
from haystack import logging
from haystack.components.generators.utils import _normalize_messages
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ComponentInfo, StreamingCallbackT, StreamingChunk, select_streaming_callback
from haystack.dataclasses.chat_message import ChatMessage
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, serialize_callable
from pydantic import BaseModel

from haystack_integrations.common.google_genai.utils import _get_client
from haystack_integrations.components.generators.google_genai.chat.interactions_utils import (
    _aggregate_streaming_chunks,
    _convert_interaction_event_to_streaming_chunk,
    _convert_interaction_to_chat_message,
    _convert_messages_to_interaction_input,
    _convert_tools_to_interactions_format,
    _process_response_format,
)

logger = logging.getLogger(__name__)


@component
class GoogleGenAIInteractionsChatGenerator:
    """
    Generates chat completions with Gemini models through the Gemini Interactions API.

    The [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) is Google's successor to the
    `generateContent` API used by `GoogleGenAIChatGenerator`. Some models and quotas, such as the free tier of newer
    Flash models, are only available through it. It supports Gemini 2.5 and newer models.

    The component is stateless like other Haystack Chat Generators: every call sends the full chat history.
    Thought signatures are kept in `ReasoningContent` and sent back automatically, so multi-turn conversations with
    thinking and tools work when you pass previous replies back in the history.
    Server-side conversation state (`previous_interaction_id`) is currently not supported.

    By default, the Interactions API stores interactions on Google's servers (1 day on the free tier,
    55 days on paid tiers). To opt out, pass `generation_kwargs={"store": False}`.

    ### Authentication
    **Gemini Developer API**: Set the `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable.
    **Vertex AI**: Use `api="vertex"` with Application Default Credentials or an API key.

    ### Usage example

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.google_genai import GoogleGenAIInteractionsChatGenerator

    chat_generator = GoogleGenAIInteractionsChatGenerator(
        model="gemini-3.8-flash",
        generation_kwargs={"generation_config": {"thinking_level": "low", "thinking_summaries": "auto"}},
    )

    response = chat_generator.run(messages=[ChatMessage.from_user("Tell me about the future of AI")])
    reply = response["replies"][0]
    print(reply.text)
    if reply.reasoning:
        print("Reasoning:", reply.reasoning.reasoning_text)
    ```

    ### Usage example with tools and an Agent

    ```python
    from haystack.components.agents import Agent
    from haystack.dataclasses import ChatMessage
    from haystack.tools import tool
    from haystack_integrations.components.generators.google_genai import GoogleGenAIInteractionsChatGenerator

    @tool
    def weather(city: str) -> str:
        \"\"\"Get the weather for a city.\"\"\"
        return f"The weather in {city} is sunny and 25°C"

    agent = Agent(chat_generator=GoogleGenAIInteractionsChatGenerator(model="gemini-3.8-flash"), tools=[weather])
    result = agent.run(messages=[ChatMessage.from_user("What's the weather in Paris?")])
    print(result["last_message"].text)
    ```

    ### Usage example with structured output

    ```python
    from pydantic import BaseModel
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.google_genai import GoogleGenAIInteractionsChatGenerator

    class City(BaseModel):
        name: str
        country: str
        population: int

    chat_generator = GoogleGenAIInteractionsChatGenerator(generation_kwargs={"response_format": City})
    response = chat_generator.run(messages=[ChatMessage.from_user("Tell me about Paris")])
    print(response["replies"][0].text) # JSON output matching the City schema
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    """A non-exhaustive list of chat models supported by this component.

    See https://aistudio.google.com/docs/interactions-overview#supported-models-agents for the models available
    through the Interactions API.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: str | None = None,
        vertex_ai_location: str | None = None,
        model: str = "gemini-3.8-flash",
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Initialize a GoogleGenAIInteractionsChatGenerator instance.

        :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
            Not needed if using Vertex AI with Application Default Credentials.
            Go to [Google AI Studio](https://aistudio.google.com/app/api-keys) for a Gemini API key.
        :param api: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
        :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
            Application Default Credentials.
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
            Required when using Vertex AI with Application Default Credentials.
        :param model: Name of the model to use.
        :param generation_kwargs: Keyword arguments passed as-is to `interactions.create`, for example
            `generation_config` (`temperature`, `max_output_tokens`, `thinking_level`, `thinking_summaries`,
            `tool_choice`, ...), `safety_settings`, `store`, or `service_tier`.
            See the [Interactions API Documentation](https://ai.google.dev/api/interactions-api) for all options.
            Set `generation_config={"thinking_summaries": "auto"}` to receive reasoning summaries.
            `response_format` also accepts a Pydantic model class or a JSON schema dict for structured output.
            `tools` declared here in the Interactions format, such as built-in Google tools, are sent along with
            `tools`; the steps they produce are not converted into the reply.
        :param streaming_callback: A callback function that is called when a new chunk is received from the stream.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param timeout: Timeout for Google GenAI client calls. If not set, the Google GenAI client default is used.
        :param max_retries: Maximum number of retries for failed requests. If not set, the Google GenAI client
            default is used.
        """
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        self._api_key = api_key
        self._api = api
        self._vertex_ai_project = vertex_ai_project
        self._vertex_ai_location = vertex_ai_location
        self._model = model
        self._generation_kwargs = generation_kwargs or {}
        self._streaming_callback = streaming_callback
        self._tools = tools
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Client | None = None
        self._async_client: AsyncClient | None = None

    def warm_up(self) -> None:
        """Create the synchronous Google Gen AI client."""
        if self._client is None:
            self._client = _get_client(
                api_key=self._api_key,
                api=self._api,
                vertex_ai_project=self._vertex_ai_project,
                vertex_ai_location=self._vertex_ai_location,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )

    async def warm_up_async(self) -> None:
        """Create the asynchronous Google Gen AI client."""
        if self._async_client is None:
            self._async_client = _get_client(
                api_key=self._api_key,
                api=self._api,
                vertex_ai_project=self._vertex_ai_project,
                vertex_ai_location=self._vertex_ai_location,
                timeout=self._timeout,
                max_retries=self._max_retries,
                async_client=True,
            )

    def close(self) -> None:
        """Close the synchronous Google Gen AI client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the asynchronous Google Gen AI client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        callback_name = serialize_callable(self._streaming_callback) if self._streaming_callback else None
        serialized_tools = serialize_tools_or_toolset(self._tools) if self._tools else None

        generation_kwargs = self._generation_kwargs.copy()
        response_format = generation_kwargs.get("response_format")
        if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            generation_kwargs["response_format"] = response_format.model_json_schema()

        return default_to_dict(
            self,
            api_key=self._api_key,
            api=self._api,
            vertex_ai_project=self._vertex_ai_project,
            vertex_ai_location=self._vertex_ai_location,
            model=self._model,
            generation_kwargs=generation_kwargs,
            streaming_callback=callback_name,
            tools=serialized_tools,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleGenAIInteractionsChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])
        return default_from_dict(cls, data)

    def _prepare_request(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None,
        tools: ToolsType | None,
    ) -> dict[str, Any]:
        """
        Build the keyword arguments for `interactions.create`.

        :param messages: The input messages.
        :param generation_kwargs: Runtime generation kwargs, merged per key over the init ones.
        :param tools: Runtime tools, replacing the init ones if provided.
        :returns: The request keyword arguments, without `stream`.
        """
        runtime_kwargs = generation_kwargs or {}
        request = {**self._generation_kwargs, **runtime_kwargs}
        init_config = self._generation_kwargs.get("generation_config")
        runtime_config = runtime_kwargs.get("generation_config")
        if isinstance(init_config, dict) and isinstance(runtime_config, dict):
            request["generation_config"] = {**init_config, **runtime_config}
        request = _process_response_format(request)

        tools = tools or self._tools
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))
        if tools:
            request["tools"] = [*request.get("tools", []), *_convert_tools_to_interactions_format(tools)]

        system_instruction, input_steps = _convert_messages_to_interaction_input(_normalize_messages(messages))
        if system_instruction:
            request["system_instruction"] = system_instruction

        request["model"] = self._model
        request["input"] = input_steps
        return request

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the chat generator on the given messages.

        :param messages: A list of ChatMessage instances representing the input messages.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param generation_kwargs: Keyword arguments for `interactions.create`. They are merged per key with the
            `generation_kwargs` passed during initialization, and `generation_config` is merged per key as well.
            Keys provided here take precedence.
        :param streaming_callback: A callback function that is called when a new chunk is received from the stream.
            If provided, it overrides the callback set during initialization.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If provided, it overrides the tools set during initialization.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated ChatMessage.
        :raises ValueError: If a message has an unsupported role or content, or tool names are duplicated.
        :raises TypeError: If `response_format` has an unsupported type.
        :raises RuntimeError: If the Interactions API call fails or the interaction ends with status `failed`.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101

        request = self._prepare_request(messages, generation_kwargs, tools)
        streaming_callback = select_streaming_callback(
            init_callback=self._streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        try:
            if streaming_callback is None:
                interaction = self._client.interactions.create(**request)
                return {"replies": [_convert_interaction_to_chat_message(interaction, self._model)]}

            component_info = ComponentInfo.from_component(self)
            chunks: list[StreamingChunk] = []
            for event in self._client.interactions.create(**request, stream=True):
                chunk = _convert_interaction_event_to_streaming_chunk(event, component_info, self._model)
                chunks.append(chunk)
                streaming_callback(chunk)
            return {"replies": [_aggregate_streaming_chunks(chunks, self._model)]}
        except Exception as e:
            msg = f"Error in Google GenAI Interactions chat generation: {e}"
            raise RuntimeError(msg) from e

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Asynchronously run the chat generator on the given messages.

        :param messages: A list of ChatMessage instances representing the input messages.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param generation_kwargs: Keyword arguments for `interactions.create`. They are merged per key with the
            `generation_kwargs` passed during initialization, and `generation_config` is merged per key as well.
            Keys provided here take precedence.
        :param streaming_callback: A callback function that is called when a new chunk is received from the stream.
            If provided, it overrides the callback set during initialization. Both sync and async callbacks work.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If provided, it overrides the tools set during initialization.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated ChatMessage.
        :raises ValueError: If a message has an unsupported role or content, or tool names are duplicated.
        :raises TypeError: If `response_format` has an unsupported type.
        :raises RuntimeError: If the Interactions API call fails or the interaction ends with status `failed`.
        """
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101

        request = self._prepare_request(messages, generation_kwargs, tools)
        streaming_callback = select_streaming_callback(
            init_callback=self._streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        try:
            if streaming_callback is None:
                interaction = await self._async_client.interactions.create(**request)
                return {"replies": [_convert_interaction_to_chat_message(interaction, self._model)]}

            component_info = ComponentInfo.from_component(self)
            chunks: list[StreamingChunk] = []
            stream = await self._async_client.interactions.create(**request, stream=True)
            async for event in stream:
                chunk = _convert_interaction_event_to_streaming_chunk(event, component_info, self._model)
                chunks.append(chunk)
                # sync callbacks are allowed in async contexts with Haystack >= 3.0, so only await async ones
                callback_result = streaming_callback(chunk)
                if inspect.isawaitable(callback_result):
                    await callback_result
            return {"replies": [_aggregate_streaming_chunks(chunks, self._model)]}
        except Exception as e:
            msg = f"Error in async Google GenAI Interactions chat generation: {e}"
            raise RuntimeError(msg) from e
