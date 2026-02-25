# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from google.genai import types
from haystack import logging
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import AsyncStreamingCallbackT, ComponentInfo, StreamingCallbackT, select_streaming_callback
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

from haystack_integrations.components.common.google_genai.utils import _get_client
from haystack_integrations.components.generators.google_genai.chat.utils import (
    _aggregate_streaming_chunks_with_reasoning,
    _convert_google_chunk_to_streaming_chunk,
    _convert_google_genai_response_to_chatmessage,
    _convert_message_to_google_genai_format,
    _convert_tools_to_google_genai_format,
    _process_thinking_config,
)

logger = logging.getLogger(__name__)


@component
class GoogleGenAIChatGenerator:
    """
    A component for generating chat completions using Google's Gemini models via the Google Gen AI SDK.

    Supports models like gemini-2.5-flash and other Gemini variants. For Gemini 2.5 series models,
    enables thinking features via `generation_kwargs={"thinking_budget": value}`.

    ### Thinking Support (Gemini 2.5 Series)
    - **Reasoning transparency**: Models can show their reasoning process
    - **Thought signatures**: Maintains thought context across multi-turn conversations with tools
    - **Configurable thinking budgets**: Control token allocation for reasoning

    Configure thinking behavior:
    - `thinking_budget: -1`: Dynamic allocation (default)
    - `thinking_budget: 0`: Disable thinking (Flash/Flash-Lite only)
    - `thinking_budget: N`: Set explicit token budget

    ### Multi-Turn Thinking with Thought Signatures
    Gemini uses **thought signatures** when tools are present - encrypted "save states" that maintain
    context across turns. Include previous assistant responses in chat history for context preservation.

    ### Authentication
    **Gemini Developer API**: Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable
    **Vertex AI**: Use `api="vertex"` with Application Default Credentials or API key

    ### Authentication Examples

    **1. Gemini Developer API (API Key Authentication)**
    ```python
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    chat_generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash")
    ```

    **2. Vertex AI (Application Default Credentials)**
    ```python
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    # Using Application Default Credentials (requires gcloud auth setup)
    chat_generator = GoogleGenAIChatGenerator(
        api="vertex",
        vertex_ai_project="my-project",
        vertex_ai_location="us-central1",
        model="gemini-2.5-flash",
    )
    ```

    **3. Vertex AI (API Key Authentication)**
    ```python
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    chat_generator = GoogleGenAIChatGenerator(
        api="vertex",
        model="gemini-2.5-flash",
    )
    ```

    ### Usage example

    ```python
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack.tools import Tool, Toolset
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    # Initialize the chat generator with thinking support
    chat_generator = GoogleGenAIChatGenerator(
        model="gemini-2.5-flash",
        generation_kwargs={"thinking_budget": 1024}  # Enable thinking with 1024 token budget
    )

    # Generate a response
    messages = [ChatMessage.from_user("Tell me about the future of AI")]
    response = chat_generator.run(messages=messages)
    print(response["replies"][0].text)

    # Access reasoning content if available
    message = response["replies"][0]
    if message.reasonings:
        for reasoning in message.reasonings:
            print("Reasoning:", reasoning.reasoning_text)

    # Tool usage example with thinking
    def weather_function(city: str):
        return f"The weather in {city} is sunny and 25°C"

    weather_tool = Tool(
        name="weather",
        description="Get weather information for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather_function
    )

    # Can use either List[Tool] or Toolset
    chat_generator_with_tools = GoogleGenAIChatGenerator(
        model="gemini-2.5-flash",
        tools=[weather_tool],  # or tools=Toolset([weather_tool])
        generation_kwargs={"thinking_budget": -1}  # Dynamic thinking allocation
    )

    messages = [ChatMessage.from_user("What's the weather in Paris?")]
    response = chat_generator_with_tools.run(messages=messages)
    ```

    ### Usage example with FileContent embedded in a ChatMessage

    ```python
    from haystack.dataclasses import ChatMessage, FileContent
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    file_content = FileContent.from_url("https://arxiv.org/pdf/2309.08632")
    chat_message = ChatMessage.from_user(content_parts=[file_content, "Summarize this paper in 100 words."])
    chat_generator = GoogleGenAIChatGenerator()
    response = chat_generator.run(messages=[chat_message])
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: str | None = None,
        vertex_ai_location: str | None = None,
        model: str = "gemini-2.5-flash",
        generation_kwargs: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ):
        """
        Initialize a GoogleGenAIChatGenerator instance.

        :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
            Not needed if using Vertex AI with Application Default Credentials.
            Go to https://aistudio.google.com/app/apikey for a Gemini API key.
            Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
        :param api: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
        :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
            Application Default Credentials.
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
            Required when using Vertex AI with Application Default Credentials.
        :param model: Name of the model to use (e.g., "gemini-2.5-flash")
        :param generation_kwargs: Configuration for generation (temperature, max_tokens, etc.).
            For Gemini 2.5 series, supports `thinking_budget` to configure thinking behavior:
            - `thinking_budget`: int, controls thinking token allocation
              - `-1`: Dynamic (default for most models)
              - `0`: Disable thinking (Flash/Flash-Lite only)
              - Positive integer: Set explicit budget
            For Gemini 3 series and newer, supports `thinking_level` to configure thinking depth:
            - `thinking_level`: str, controls thinking (https://ai.google.dev/gemini-api/docs/thinking#levels-budgets)
              - `minimal`: Matches the "no thinking" setting for most queries. The model may think very minimally for
                    complex coding tasks. Minimizes latency for chat or high throughput applications.
              - `low`: Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput
                    applications.
              - `medium`: Balanced thinking for most tasks.
              - `high`: (Default, dynamic): Maximizes reasoning depth. The model may take significantly longer to reach
                    a first token, but the output will be more carefully reasoned.
        :param safety_settings: Safety settings for content filtering
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        """
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        self._client = _get_client(
            api_key=api_key,
            api=api,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
        )

        self._api_key = api_key
        self._api = api
        self._vertex_ai_project = vertex_ai_project
        self._vertex_ai_location = vertex_ai_location
        self._model = model
        self._generation_kwargs = generation_kwargs or {}
        self._safety_settings = safety_settings or []
        self._streaming_callback = streaming_callback
        self._tools = tools

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        callback_name = serialize_callable(self._streaming_callback) if self._streaming_callback else None
        serialized_tools = serialize_tools_or_toolset(self._tools) if self._tools else None
        return default_to_dict(
            self,
            api_key=self._api_key.to_dict(),
            api=self._api,
            vertex_ai_project=self._vertex_ai_project,
            vertex_ai_location=self._vertex_ai_location,
            model=self._model,
            generation_kwargs=self._generation_kwargs,
            safety_settings=self._safety_settings,
            streaming_callback=callback_name,
            tools=serialized_tools,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleGenAIChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        if "streaming_callback" in init_params and init_params["streaming_callback"] is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])
        return default_from_dict(cls, data)

    def _handle_streaming_response(
        self, response_stream: Iterator[types.GenerateContentResponse], streaming_callback: StreamingCallbackT
    ) -> dict[str, list[ChatMessage]]:
        """
        Handle streaming response from Google Gen AI generate_content_stream.
        :param response_stream: The streaming response from generate_content_stream.
        :param streaming_callback: The callback function for streaming chunks.
        :returns: A dictionary with the replies.
        """
        component_info = ComponentInfo.from_component(self)

        try:
            chunks = []

            for i, chunk in enumerate(response_stream):
                streaming_chunk = _convert_google_chunk_to_streaming_chunk(
                    chunk=chunk, index=i, component_info=component_info, model=self._model
                )
                chunks.append(streaming_chunk)

                # Stream the chunk
                streaming_callback(streaming_chunk)

            # Use custom aggregation that supports reasoning content
            message = _aggregate_streaming_chunks_with_reasoning(chunks)
            return {"replies": [message]}

        except Exception as e:
            msg = f"Error in streaming response: {e}"
            raise RuntimeError(msg) from e

    async def _handle_streaming_response_async(
        self, response_stream: AsyncIterator[types.GenerateContentResponse], streaming_callback: AsyncStreamingCallbackT
    ) -> dict[str, list[ChatMessage]]:
        """
        Handle async streaming response from Google Gen AI generate_content_stream.
        :param response_stream: The async streaming response from generate_content_stream.
        :param streaming_callback: The async callback function for streaming chunks.
        :returns: A dictionary with the replies.
        """
        component_info = ComponentInfo.from_component(self)

        try:
            chunks = []

            i = 0
            chunk = None
            async for chunk in response_stream:
                i += 1

                streaming_chunk = _convert_google_chunk_to_streaming_chunk(
                    chunk=chunk, index=i, component_info=component_info, model=self._model
                )
                chunks.append(streaming_chunk)

                # Stream the chunk
                await streaming_callback(streaming_chunk)

            # Use custom aggregation that supports reasoning content
            message = _aggregate_streaming_chunks_with_reasoning(chunks)
            return {"replies": [message]}

        except Exception as e:
            msg = f"Error in async streaming response: {e}"
            raise RuntimeError(msg) from e

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, Any]:
        """
        Run the Google Gen AI chat generator on the given input data.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Configuration for generation. If provided, it will override
        the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
        :param safety_settings: Safety settings for content filtering. If provided, it will override the
        default settings.
        :param streaming_callback: A callback function that is called when a new token is
        received from the stream.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
        If provided, it will override the tools set during initialization.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated ChatMessage responses.

        :raises RuntimeError: If there is an error in the Google Gen AI chat generation.
        :raises ValueError: If a ChatMessage does not contain at least one of TextContent, ToolCall, or
        ToolCallResult or if the role in ChatMessage is different from User, System, Assistant.
        """
        # Use provided configs or fall back to instance defaults
        generation_kwargs = generation_kwargs or self._generation_kwargs
        safety_settings = safety_settings or self._safety_settings
        tools = tools or self._tools

        # Process thinking configuration
        generation_kwargs = _process_thinking_config(generation_kwargs)

        # Select appropriate streaming callback
        streaming_callback = select_streaming_callback(
            init_callback=self._streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=False,
        )

        # Check for duplicate tool names
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        # Handle system message if present
        system_instruction = None
        chat_messages = messages

        if messages and messages[0].is_from(ChatRole.SYSTEM):
            system_instruction = messages[0].text or ""
            chat_messages = messages[1:]

        # Convert messages to Google Gen AI Content format
        contents: list[types.ContentUnionDict] = []
        for msg in chat_messages:
            contents.append(_convert_message_to_google_genai_format(msg))

        try:
            # Prepare generation config
            config_params = generation_kwargs.copy() if generation_kwargs else {}
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            # Add safety settings if provided
            if safety_settings:
                config_params["safety_settings"] = safety_settings

            # Add tools if provided
            if tools:
                config_params["tools"] = _convert_tools_to_google_genai_format(tools)

            config = types.GenerateContentConfig(**config_params) if config_params else None

            if streaming_callback:
                # Use streaming
                response_stream = self._client.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                return self._handle_streaming_response(response_stream, streaming_callback)
            else:
                # Use non-streaming
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )

                reply = _convert_google_genai_response_to_chatmessage(response, self._model)
                return {"replies": [reply]}

        except Exception as e:
            # Check if the error is related to thinking configuration
            error_str = str(e).lower()
            if ("thinking" in error_str or "thinking_config" in error_str) and "thinking_config" in config_params:
                # Provide a more helpful error message for thinking configuration issues
                error_msg = (
                    f"Thinking configuration error for model '{self._model}': {e}\n"
                    f"The model may not support thinking features or the thinking_budget value may be invalid. "
                    f"Try removing the 'thinking_budget' parameter from generation_kwargs or use a different model."
                )
                raise RuntimeError(error_msg) from e

            error_msg = f"Error in Google Gen AI chat generation: {e}"
            raise RuntimeError(error_msg) from e

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, Any]:
        """
        Async version of the run method. Run the Google Gen AI chat generator on the given input data.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Configuration for generation. If provided, it will override
        the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
        See https://ai.google.dev/gemini-api/docs/thinking for possible values.
        :param safety_settings: Safety settings for content filtering. If provided, it will override the
        default settings.
        :param streaming_callback: A callback function that is called when a new token is
        received from the stream.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
        If provided, it will override the tools set during initialization.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated ChatMessage responses.

        :raises RuntimeError: If there is an error in the async Google Gen AI chat generation.
        :raises ValueError: If a ChatMessage does not contain at least one of TextContent, ToolCall, or
        ToolCallResult or if the role in ChatMessage is different from User, System, Assistant.
        """
        # Use provided configs or fall back to instance defaults
        generation_kwargs = generation_kwargs or self._generation_kwargs
        safety_settings = safety_settings or self._safety_settings
        tools = tools or self._tools

        # Process thinking configuration
        generation_kwargs = _process_thinking_config(generation_kwargs)

        # Select appropriate streaming callback
        streaming_callback = select_streaming_callback(
            init_callback=self._streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=True,
        )

        # Check for duplicate tool names
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        # Handle system message if present
        system_instruction = None
        chat_messages = messages

        if messages and messages[0].is_from(ChatRole.SYSTEM):
            system_instruction = messages[0].text or ""
            chat_messages = messages[1:]

        # Convert messages to Google Gen AI Content format
        contents: list[types.ContentUnion] = []
        for msg in chat_messages:
            contents.append(_convert_message_to_google_genai_format(msg))

        try:
            # Prepare generation config
            config_params = generation_kwargs.copy() if generation_kwargs else {}
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            # Add safety settings if provided
            if safety_settings:
                config_params["safety_settings"] = safety_settings

            # Add tools if provided
            if tools:
                config_params["tools"] = _convert_tools_to_google_genai_format(tools)

            config = types.GenerateContentConfig(**config_params) if config_params else None

            if streaming_callback:
                # Use streaming
                response_stream = await self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                return await self._handle_streaming_response_async(response_stream, streaming_callback)
            else:
                # Use non-streaming
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                reply = _convert_google_genai_response_to_chatmessage(response, self._model)
                return {"replies": [reply]}

        except Exception as e:
            # Check if the error is related to thinking configuration
            error_str = str(e).lower()
            if ("thinking" in error_str or "thinking_config" in error_str) and "thinking_config" in config_params:
                # Provide a more helpful error message for thinking configuration issues
                error_msg = (
                    f"Thinking configuration error for model '{self._model}': {e}\n"
                    f"The model may not support thinking features or the thinking_budget value may be invalid. "
                    f"Try removing the 'thinking_budget' parameter from generation_kwargs or use a different model."
                )
                raise RuntimeError(error_msg) from e

            error_msg = f"Error in async Google Gen AI chat generation: {e}"
            raise RuntimeError(error_msg) from e
