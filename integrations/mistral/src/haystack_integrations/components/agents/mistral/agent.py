# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import (
    ChatMessage,
    StreamingCallbackT,
    StreamingChunk,
    ToolCall,
)
from haystack.lazy_imports import LazyImport
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

with LazyImport(message="Run 'pip install mistralai'") as mistralai_import:
    from mistralai import Mistral, models

logger = logging.getLogger(__name__)

ToolChoiceType = Union[
    Literal["auto", "none", "any", "required"],
    dict[str, Any],
]


@component
class MistralAgent:
    """
    Generates text using Mistral AI Agents via the official Mistral Python SDK.

    NOTE:
    If you get the error message:

        "Cannot set function calling tools in the request and have tools in the agent"

    This is a Mistral API limitation - if your agent in the Mistral console already has tools configured, you cannot
    pass additional tools in the API request. They are mutually exclusive.

    For more information on Mistral Agents, see:
    [Mistral Agents API](https://docs.mistral.ai/api/endpoint/agents)

    Usage example:
    ```python
    from haystack_integrations.components.agents.mistral import MistralAgent
    from haystack.dataclasses import ChatMessage

    # Initialize with your agent ID from the Mistral console
    agent = MistralAgent(agent_id="your-agent-id")

    messages = [ChatMessage.from_user("What can you help me with?")]
    response = agent.run(messages)
    print(response["replies"][0].text)
    ```

    Streaming example:
    ```python
    def my_callback(chunk):
        print(chunk.content, end="", flush=True)

    agent = MistralAgent(
        agent_id="your-agent-id",
        streaming_callback=my_callback
    )
    response = agent.run([ChatMessage.from_user("Tell me a story")])
    ```
    """

    def __init__(
        self,
        agent_id: str,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[ToolsType] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        parallel_tool_calls: bool = True,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        timeout_ms: Optional[int] = 30000,
    ):
        """
        Creates an instance of MistralAgent.

        :param agent_id:
            The ID of the Mistral Agent to use. Required. Get this from the
            Mistral AI console after creating an agent.
        :param api_key:
            The Mistral API key. Defaults to environment variable `MISTRAL_API_KEY`.
        :param streaming_callback:
            A callback function called when a new token is received from the stream.
        :param tools:
            Additional tools the agent can use beyond its pre-configured tools.
            A list of Tool and/or Toolset objects.
        :param tool_choice:
            Controls which tool is called. Options:
            - "auto": Model decides whether to use tools
            - "none": No tools, generate text only
            - "any" or "required": Must call one or more tools
            - {"type": "function", "function": {"name": "..."}}: Force specific tool
        :param parallel_tool_calls:
            Whether to enable parallel function calling. Defaults to True.
        :param generation_kwargs:
            Additional parameters for the API call. Supported parameters:
            - `max_tokens`: Maximum tokens to generate
            - `frequency_penalty`: Penalize word repetition (default: 0)
            - `presence_penalty`: Encourage vocabulary diversity (default: 0)
            - `n`: Number of completions to return
            - `random_seed`: Seed for deterministic results
            - `stop`: Stop sequences
            - `response_format`: Output format (text/json_object/json_schema)
            - `prediction`: Expected completion for optimization
            - `prompt_mode`: Set to "reasoning" for reasoning models
        :param timeout_ms:
            Request timeout in milliseconds. Defaults to 30000 (30 seconds).
        """
        self.agent_id = agent_id
        self.api_key = api_key
        self.streaming_callback = streaming_callback
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.generation_kwargs = generation_kwargs or {}
        self.timeout_ms = timeout_ms

        _check_duplicate_tool_names(flatten_tools_or_toolsets(self.tools))

        self._client = None
        self._async_client = None

    def warm_up(self):
        if self._client:
            return
        mistralai_import.check()
        self._client = Mistral(api_key=self.api_key.resolve_value(),timeout_ms=self.timeout_ms)

    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:

        sdk_messages = []

        for msg in messages:
            # OpenAI format is compatible with Mistral
            openai_format = msg.to_openai_dict_format()

            # Ensure content is a string (not a list of content blocks)
            content = openai_format.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "".join(text_parts)

            sdk_message = {
                "role": openai_format.get("role", "user"),
                "content": content,
            }

            # Include tool_call_id for tool messages
            if openai_format.get("tool_call_id"):
                sdk_message["tool_call_id"] = openai_format["tool_call_id"]

            # Include tool_calls for assistant messages
            if openai_format.get("tool_calls"):
                sdk_message["tool_calls"] = openai_format["tool_calls"]

            sdk_messages.append(sdk_message)

        return sdk_messages

    def _build_tools(self, tools: Optional[ToolsType] = None) -> Optional[list[dict[str, Any]]]:
        """Convert Haystack tools to Mistral format."""
        flattened_tools = flatten_tools_or_toolsets(tools or self.tools)
        if not flattened_tools:
            return None
        return [
            {"type": "function", "function": tool.tool_spec}
            for tool in flattened_tools
        ]

    @staticmethod
    def _parse_response(response: Any) -> list[ChatMessage]:
        """
        Parse the Mistral response into Haystack ChatMessages.

        :param response: The response from mistral.agents.complete()
        :returns:
            List of ChatMessage objects
        """
        messages = []

        for choice in response.choices:
            message = choice.message
            content = message.content or ""

            # Parse tool calls if present
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc.type == "function":
                        try:
                            arguments = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        tool_calls.append(
                            ToolCall(
                                id=tc.id,
                                tool_name=tc.function.name,
                                arguments=arguments,
                            )
                        )

            # Build metadata
            meta = {
                "model": response.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            }

            chat_message = ChatMessage.from_assistant(
                text=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
                meta=meta,
            )
            messages.append(chat_message)

        return messages

    @staticmethod
    def _handle_streaming(stream_response: Any, callback: StreamingCallbackT,) -> list[ChatMessage]:
        """
        Handle streaming response from the Mistral SDK.

        :param stream_response: The streaming response iterator
        :param callback: The callback to invoke for each chunk
        :returns:
            List containing the final assembled ChatMessage
        """
        collected_content = ""
        collected_tool_calls: dict[int, dict] = {}
        meta: dict[str, Any] = {}

        for chunk in stream_response:
            # Extract metadata from response (model is on chunk.data, not chunk)
            if not meta and chunk.data.model:
                meta["model"] = chunk.data.model

            for choice in chunk.data.choices:
                delta = choice.delta

                # Handle text content
                if delta.content:
                    collected_content += delta.content
                    streaming_chunk = StreamingChunk(
                        content=delta.content,
                        meta={
                            "model": chunk.data.model,
                            "index": choice.index,
                            "finish_reason": choice.finish_reason,
                        },
                    )
                    callback(streaming_chunk)

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, "index") else 0
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            collected_tool_calls[idx]["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                collected_tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[idx]["arguments"] += tc.function.arguments

                # Capture finish reason
                if choice.finish_reason:
                    meta["finish_reason"] = choice.finish_reason
                    meta["index"] = choice.index

            # Capture usage from final chunk
            if chunk.data.usage:
                meta["usage"] = {
                    "prompt_tokens": chunk.data.usage.prompt_tokens,
                    "completion_tokens": chunk.data.usage.completion_tokens,
                    "total_tokens": chunk.data.usage.total_tokens,
                }

        # Build final tool calls
        tool_calls = []
        for idx in sorted(collected_tool_calls.keys()):
            tc_data = collected_tool_calls[idx]
            try:
                arguments = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    tool_name=tc_data["name"],
                    arguments=arguments,
                )
            )

        # Create final message
        chat_message = ChatMessage.from_assistant(
            text=collected_content if collected_content else None,
            tool_calls=tool_calls if tool_calls else None,
            meta=meta,
        )

        return [chat_message]

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[ToolsType] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Invoke the Mistral Agent with the provided messages.

        :param messages:
            A list of ChatMessage instances representing the conversation.
        :param streaming_callback:
            A callback function for streaming. Overrides the init callback.
        :param tools:
            Additional tools for this request. Overrides init tools.
        :param tool_choice:
            Tool choice for this request. Overrides init tool_choice.
        :param generation_kwargs:
            Additional generation parameters. Merged with init params.

        :returns:
            A dictionary with key `replies` containing a list of ChatMessage responses.
        """
        self.warm_up()

        if not messages:
            return {"replies": []}

        # Select streaming callback
        effective_callback = streaming_callback or self.streaming_callback

        # Convert messages
        sdk_messages = MistralAgent._convert_messages(messages)

        # Merge generation kwargs
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "agent_id": self.agent_id,
            "messages": sdk_messages,
        }

        # Add tools if provided
        sdk_tools = self._build_tools(tools)
        if sdk_tools:
            request_kwargs["tools"] = sdk_tools
            request_kwargs["parallel_tool_calls"] = self.parallel_tool_calls

            # Add tool_choice only when tools are present
            effective_tool_choice = tool_choice or self.tool_choice
            if effective_tool_choice:
                request_kwargs["tool_choice"] = effective_tool_choice

        # Add generation kwargs
        for key, value in merged_kwargs.items():
            if value is not None:
                request_kwargs[key] = value

        try:
            if effective_callback:
                # Streaming request
                request_kwargs["stream"] = True
                stream_response = self._client.agents.stream(**request_kwargs)
                replies = MistralAgent._handle_streaming(stream_response, effective_callback)
            else:
                # Non-streaming request
                response = self._client.agents.complete(**request_kwargs)
                replies = self._parse_response(response)

            return {"replies": replies}

        except Exception as e:
            if isinstance(e, models.HTTPValidationError):
                msg = "Mistral validation error: {detail}"
                logger.error(msg, detail=e.data.detail if hasattr(e, "data") else str(e))
                error_msg = f"Mistral validation error: {e}"
                raise ValueError(error_msg) from e

            elif isinstance(e, models.MistralError):
                msg = "Mistral API error: {status_code} - {message}"
                logger.error(msg, status_code=e.status_code, message=e.message)
                error_msg = f"Mistral API error ({e.status_code}): {e.message}"
                raise ValueError(error_msg) from e

            raise


    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[ToolsType] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Asynchronously invoke the Mistral Agent.

        Same parameters as `run()`.
        """
        self.warm_up()

        if not messages:
            return {"replies": []}

        effective_callback = streaming_callback or self.streaming_callback
        sdk_messages = self._convert_messages(messages)
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        request_kwargs: dict[str, Any] = {
            "agent_id": self.agent_id,
            "messages": sdk_messages,
        }

        sdk_tools = self._build_tools(tools)
        if sdk_tools:
            request_kwargs["tools"] = sdk_tools
            request_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
            effective_tool_choice = tool_choice or self.tool_choice
            if effective_tool_choice:
                request_kwargs["tool_choice"] = effective_tool_choice

        for key, value in merged_kwargs.items():
            if value is not None:
                request_kwargs[key] = value

        try:
            if effective_callback:
                # Async streaming
                request_kwargs["stream"] = True
                stream_response = await self._client.agents.stream_async(**request_kwargs)
                # Note: For full async streaming, we'd need an async callback
                # This is a simplified version
                replies = await Mistral._handle_async_streaming(stream_response, effective_callback)
            else:
                response = await self._client.agents.complete_async(**request_kwargs)
                replies = self._parse_response(response)

            return {"replies": replies}

        except Exception as e:
            if isinstance(e, (models.HTTPValidationError, models.MistralError)):
                error_msg = f"Mistral API error: {e}"
                raise ValueError(error_msg) from e
            raise

    @staticmethod
    async def _handle_async_streaming(
        stream_response: Any,
        callback: StreamingCallbackT,
    ) -> list[ChatMessage]:
        """Handle async streaming response."""
        collected_content = ""
        collected_tool_calls: dict[int, dict] = {}
        meta: dict[str, Any] = {}

        async for chunk in stream_response:
            if not meta and chunk.data.model:
                meta["model"] = chunk.data.model

            for choice in chunk.data.choices:
                delta = choice.delta

                if delta.content:
                    collected_content += delta.content
                    streaming_chunk = StreamingChunk(
                        content=delta.content,
                        meta={
                            "model": chunk.data.model,
                            "index": choice.index,
                            "finish_reason": choice.finish_reason,
                        },
                    )
                    # For async streaming, callback should be awaitable
                    if callable(callback):
                        result = callback(streaming_chunk)
                        if hasattr(result, "__await__"):
                            await result

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, "index") else 0
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            collected_tool_calls[idx]["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                collected_tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[idx]["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    meta["finish_reason"] = choice.finish_reason
                    meta["index"] = choice.index

            if chunk.data.usage:
                meta["usage"] = {
                    "prompt_tokens": chunk.data.usage.prompt_tokens,
                    "completion_tokens": chunk.data.usage.completion_tokens,
                    "total_tokens": chunk.data.usage.total_tokens,
                }

        tool_calls = []
        for idx in sorted(collected_tool_calls.keys()):
            tc_data = collected_tool_calls[idx]
            try:
                arguments = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc_data["id"], tool_name=tc_data["name"], arguments=arguments)
            )

        return [
            ChatMessage.from_assistant(
                text=collected_content if collected_content else None,
                tool_calls=tool_calls if tool_calls else None,
                meta=meta,
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: A dictionary representation of the component.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        return default_to_dict(
            self,
            agent_id=self.agent_id,
            api_key=self.api_key.to_dict(),
            streaming_callback=callback_name,
            tools=serialize_tools_or_toolset(self.tools),
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            generation_kwargs=self.generation_kwargs,
            timeout_ms=self.timeout_ms,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MistralAgent":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of the component.
        :returns:
            An instance of MistralAgent
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")

        init_params = data.get("init_parameters", {})
        if init_params.get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(
                init_params["streaming_callback"]
            )

        return default_from_dict(cls, data)
