# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from google.genai import types
from haystack import logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ComponentInfo,
    FinishReason,
    ImageContent,
    StreamingCallbackT,
    StreamingChunk,
    TextContent,
    ToolCall,
    ToolCallDelta,
    ToolCallResult,
    select_streaming_callback,
)
from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ReasoningContent
from haystack.tools import (
    Tool,
    Toolset,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from jsonref import replace_refs

from haystack_integrations.components.common.google_genai.utils import _get_client
from haystack_integrations.components.generators.google_genai.chat.utils import remove_key_from_schema

# Mapping from Google GenAI finish reasons to Haystack FinishReason values
FINISH_REASON_MAPPING: Dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
}


logger = logging.getLogger(__name__)

# Google AI supported image MIME types based on documentation
# https://ai.google.dev/gemini-api/docs/image-understanding?lang=python
GOOGLE_AI_SUPPORTED_MIME_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",  # Common alias
    "image/webp": "webp",
    "image/heic": "heic",
    "image/heif": "heif",
}


def _convert_message_to_google_genai_format(message: ChatMessage) -> types.Content:
    """
    Converts a Haystack ChatMessage to Google Gen AI Content format.

    :param message: The Haystack ChatMessage to convert.
    :returns: Google Gen AI Content object.
    """
    # Check if message has content
    if not message._content:
        msg = "A `ChatMessage` must contain at least one content part."
        raise ValueError(msg)

    parts = []

    # Check if this message has thought signatures from a previous response
    # These need to be reconstructed in their original part structure
    thought_signatures = message.meta.get("thought_signatures", []) if message.meta else []

    # If we have thought signatures, we need to reconstruct the exact part structure
    # from the previous assistant response to maintain multi-turn thinking context
    if thought_signatures and message.is_from(ChatRole.ASSISTANT):
        # Track which tool calls we've used (to handle multiple tool calls)
        tool_call_index = 0

        # Reconstruct parts with their original thought signatures
        for sig_info in thought_signatures:
            part_dict: Dict[str, Any] = {}

            # Check what type of content this part had
            if sig_info.get("has_text"):
                # Find the corresponding text content
                if sig_info.get("is_thought"):
                    # This was a thought part - find it in reasoning content
                    if message.reasoning:
                        part_dict["text"] = message.reasoning.reasoning_text
                        part_dict["thought"] = True
                else:
                    # Regular text part
                    part_dict["text"] = message.text or ""

            if sig_info.get("has_function_call"):
                # Find the corresponding tool call by index
                if message.tool_calls and tool_call_index < len(message.tool_calls):
                    tool_call = message.tool_calls[tool_call_index]
                    part_dict["function_call"] = types.FunctionCall(
                        id=tool_call.id, name=tool_call.tool_name, args=tool_call.arguments
                    )
                    tool_call_index += 1  # Move to next tool call for next part

            # Add the thought signature to preserve context
            part_dict["thought_signature"] = sig_info["signature"]

            parts.append(types.Part(**part_dict))

        # If we reconstructed from signatures, we're done
        if parts:
            role = "model"  # Assistant messages with signatures are always from the model
            return types.Content(role=role, parts=parts)

    # Standard processing for messages without thought signatures
    for content_part in message._content:
        if isinstance(content_part, TextContent):
            # Only add text parts that are not empty to avoid unnecessary empty text parts
            if content_part.text.strip():
                parts.append(types.Part(text=content_part.text))

        elif isinstance(content_part, ImageContent):
            if not message.is_from(ChatRole.USER):
                msg = "Image content is only supported for user messages"
                raise ValueError(msg)

            # Validate image MIME type and format
            if not content_part.mime_type:
                msg = "Image MIME type could not be determined. Please provide a valid image with detectable format."
                raise ValueError(msg)

            if content_part.mime_type not in GOOGLE_AI_SUPPORTED_MIME_TYPES:
                supported_types = list(GOOGLE_AI_SUPPORTED_MIME_TYPES.keys())
                msg = (
                    f"Unsupported image MIME type: {content_part.mime_type}. "
                    f"Google AI supports the following MIME types: {supported_types}"
                )
                raise ValueError(msg)

            # Use inline image data approach
            try:
                # ImageContent already has base64 data, decode it for bytes
                image_bytes = base64.b64decode(content_part.base64_image)

                # Create Part using from_bytes method
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=content_part.mime_type)
                parts.append(image_part)

            except Exception as e:
                msg = f"Failed to process image data: {e}"
                raise RuntimeError(msg) from e

        elif isinstance(content_part, ToolCall):
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=content_part.id, name=content_part.tool_name, args=content_part.arguments
                    )
                )
            )

        elif isinstance(content_part, ToolCallResult):
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=content_part.origin.id,
                        name=content_part.origin.tool_name,
                        response={"result": content_part.result},
                    )
                )
            )
        elif isinstance(content_part, ReasoningContent):
            # Reasoning content is for human transparency only, not for maintaining LLM context
            # Thought signatures (stored in message.meta) handle context preservation
            # Leave this here so we don't implement reasoning content handling in the future accidentally
            pass

    # Determine role
    if message.is_from(ChatRole.USER) or message.tool_call_results:
        role = "user"
    elif message.is_from(ChatRole.ASSISTANT):
        role = "model"
    elif message.is_from(ChatRole.SYSTEM):
        # System messages will be handled separately as system instruction
        # When we convert a list of ChatMessage to be sent to google genai,
        # we need to handle system messages separately as system instruction and we only take the first message
        # as the system instruction - if it is present.
        #
        # If we find any additional system messages, we will treat them as user messages
        role = "user"
    else:
        msg = f"Unsupported message role: {message._role}"
        raise ValueError(msg)

    return types.Content(role=role, parts=parts)


def _sanitize_tool_schema(tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes a tool schema to remove any keys that are not supported by Google Gen AI.

    Google Gen AI does not support additionalProperties, $schema, $defs, or $ref in the tool schema.

    :param tool_schema: The tool schema to sanitize.
    :returns: The sanitized tool schema.
    """
    # google Gemini does not support additionalProperties and $schema in the tool schema
    sanitized_schema = remove_key_from_schema(tool_schema, "additionalProperties")
    sanitized_schema = remove_key_from_schema(sanitized_schema, "$schema")
    # expand $refs in the tool schema
    expanded_schema = replace_refs(sanitized_schema)
    # and remove the $defs key leaving the rest of the schema
    final_schema = remove_key_from_schema(expanded_schema, "$defs")

    if not isinstance(final_schema, dict):
        msg = "Tool schema must be a dictionary after sanitization"
        raise ValueError(msg)

    return final_schema


def _convert_tools_to_google_genai_format(tools: Union[List[Tool], Toolset]) -> List[types.Tool]:
    """
    Converts a list of Haystack Tools or a Toolset to Google Gen AI Tool format.

    :param tools: List of Haystack Tool objects or a Toolset.
    :returns: List of Google Gen AI Tool objects.
    """
    # Convert Toolset to list if needed
    if isinstance(tools, Toolset):
        tools = list(tools)

    function_declarations: List[types.FunctionDeclaration] = []
    for tool in tools:
        parameters = _sanitize_tool_schema(tool.parameters)
        function_declarations.append(
            types.FunctionDeclaration(
                name=tool.name, description=tool.description, parameters=types.Schema(**parameters)
            )
        )

    # Return a single Tool object with all function declarations as in the Google GenAI docs
    # we could also return multiple Tool objects, doesn't seem to make a difference
    # revisit this decision
    return [types.Tool(function_declarations=function_declarations)]


def _convert_google_genai_response_to_chatmessage(response: types.GenerateContentResponse, model: str) -> ChatMessage:
    """
    Converts a Google Gen AI response to a Haystack ChatMessage.

    :param response: The response from Google Gen AI.
    :param model: The model name.
    :returns: A Haystack ChatMessage.
    """
    text_parts = []
    tool_calls = []
    reasoning_parts = []
    thought_signatures = []  # Store thought signatures for multi-turn context

    # Extract text, function calls, thoughts, and thought signatures from response
    finish_reason = None
    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if candidate.content is not None and candidate.content.parts is not None:
            for i, part in enumerate(candidate.content.parts):
                # Check for thought signature on this part
                if hasattr(part, "thought_signature") and part.thought_signature:
                    # Store the thought signature with its part index for reconstruction
                    thought_signatures.append(
                        {
                            "part_index": i,
                            "signature": part.thought_signature,
                            "has_text": part.text is not None,
                            "has_function_call": part.function_call is not None,
                            "is_thought": hasattr(part, "thought") and part.thought,
                        }
                    )

                if part.text is not None and not (hasattr(part, "thought") and part.thought):
                    text_parts.append(part.text)
                if part.function_call is not None:
                    tool_call = ToolCall(
                        tool_name=part.function_call.name or "",
                        arguments=dict(part.function_call.args) if part.function_call.args else {},
                        id=part.function_call.id,
                    )
                    tool_calls.append(tool_call)
                # Handle thought parts for Gemini 2.5 series
                if hasattr(part, "thought") and part.thought:
                    # Extract thought content
                    if part.text:
                        reasoning_parts.append(part.text)

    # Combine text parts
    text = " ".join(text_parts) if text_parts else ""

    usage_metadata = response.usage_metadata

    # Create usage metadata including thinking tokens if available
    usage = {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0),
        "total_tokens": getattr(usage_metadata, "total_token_count", 0),
    }

    # Add thinking token count if available
    if usage_metadata and hasattr(usage_metadata, "thoughts_token_count") and usage_metadata.thoughts_token_count:
        usage["thoughts_token_count"] = usage_metadata.thoughts_token_count

        # Create meta with reasoning content and thought signatures if available
    meta: Dict[str, Any] = {
        "model": model,
        "finish_reason": FINISH_REASON_MAPPING.get(finish_reason or ""),
        "usage": usage,
    }

    # Add thought signatures to meta if present (for multi-turn context preservation)
    if thought_signatures:
        meta["thought_signatures"] = thought_signatures

    # Create ReasoningContent object if there are reasoning parts
    reasoning_content = None
    if reasoning_parts:
        reasoning_text = " ".join(reasoning_parts)
        reasoning_content = ReasoningContent(reasoning_text=reasoning_text)

    # Create ChatMessage
    message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta, reasoning=reasoning_content)

    return message


@component
class GoogleGenAIChatGenerator:
    """
    A component for generating chat completions using Google's Gemini models via the Google Gen AI SDK.

    Supports models like gemini-2.0-flash and other Gemini variants. For Gemini 2.5 series models,
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
    chat_generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash")
    ```

    **2. Vertex AI (Application Default Credentials)**
    ```python
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    # Using Application Default Credentials (requires gcloud auth setup)
    chat_generator = GoogleGenAIChatGenerator(
        api="vertex",
        vertex_ai_project="my-project",
        vertex_ai_location="us-central1",
        model="gemini-2.0-flash"
    )
    ```

    **3. Vertex AI (API Key Authentication)**
    ```python
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    chat_generator = GoogleGenAIChatGenerator(
        api="vertex",
        model="gemini-2.0-flash"
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
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: Optional[str] = None,
        vertex_ai_location: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
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
        :param model: Name of the model to use (e.g., "gemini-2.0-flash")
        :param generation_kwargs: Configuration for generation (temperature, max_tokens, etc.).
            For Gemini 2.5 series, supports `thinking_budget` to configure thinking behavior:
            - `thinking_budget`: int, controls thinking token allocation
              - `-1`: Dynamic (default for most models)
              - `0`: Disable thinking (Flash/Flash-Lite only)
              - Positive integer: Set explicit budget
        :param safety_settings: Safety settings for content filtering
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should have a unique name.
        """
        _check_duplicate_tool_names(list(tools or []))  # handles Toolset as well

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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleGenAIChatGenerator":
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

    def _convert_google_chunk_to_streaming_chunk(
        self,
        chunk: types.GenerateContentResponse,
        index: int,
        component_info: ComponentInfo,
    ) -> StreamingChunk:
        """
        Convert a chunk from Google Gen AI to a Haystack StreamingChunk.

        :param chunk: The chunk from Google Gen AI.
        :param index: The index of the chunk.
        :returns: A StreamingChunk object.
        """
        content = ""
        tool_calls: List[ToolCallDelta] = []
        finish_reason = None
        reasoning_deltas: List[Dict[str, str]] = []
        thought_signature_deltas: List[Dict[str, Any]] = []  # Track thought signatures in streaming

        if chunk.candidates:
            candidate = chunk.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)

        usage_metadata = chunk.usage_metadata

        usage = {
            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
            "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
            "total_tokens": getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0,
        }

        # Add thinking token count if available
        if usage_metadata and hasattr(usage_metadata, "thoughts_token_count") and usage_metadata.thoughts_token_count:
            usage["thoughts_token_count"] = usage_metadata.thoughts_token_count

        if candidate.content and candidate.content.parts:
            tc_index = -1
            for part_index, part in enumerate(candidate.content.parts):
                # Check for thought signature on this part (for multi-turn context)
                if hasattr(part, "thought_signature") and part.thought_signature:
                    thought_signature_deltas.append(
                        {
                            "part_index": part_index,
                            "signature": part.thought_signature,
                            "has_text": part.text is not None,
                            "has_function_call": part.function_call is not None,
                            "is_thought": hasattr(part, "thought") and part.thought,
                        }
                    )

                if part.text is not None and not (hasattr(part, "thought") and part.thought):
                    content += part.text

                elif part.function_call:
                    tc_index += 1
                    tool_calls.append(
                        ToolCallDelta(
                            # Google GenAI does not provide index, but it is required for tool calls
                            index=tc_index,
                            id=part.function_call.id,
                            tool_name=part.function_call.name or "",
                            arguments=json.dumps(part.function_call.args) if part.function_call.args else None,
                        )
                    )

                # Handle thought parts for Gemini 2.5 series
                elif hasattr(part, "thought") and part.thought:
                    thought_delta = {
                        "type": "reasoning",
                        "content": part.text if part.text else "",
                    }
                    reasoning_deltas.append(thought_delta)

        # start is only used by print_streaming_chunk. We try to make a reasonable assumption here but it should not be
        # a problem if we change it in the future.
        start = index == 0 or len(tool_calls) > 0

        # Create meta with reasoning deltas and thought signatures if available
        meta: Dict[str, Any] = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "model": self._model,
            "usage": usage,
        }

        # Add reasoning deltas to meta if available
        if reasoning_deltas:
            meta["reasoning_deltas"] = reasoning_deltas

        # Add thought signature deltas to meta if available (for multi-turn context)
        if thought_signature_deltas:
            meta["thought_signature_deltas"] = thought_signature_deltas

        return StreamingChunk(
            content="" if tool_calls else content,  # prioritize tool calls over content when both are present
            tool_calls=tool_calls,
            component_info=component_info,
            index=index,
            start=start,
            finish_reason=FINISH_REASON_MAPPING.get(finish_reason or ""),
            meta=meta,
        )

    def _aggregate_streaming_chunks_with_reasoning(self, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Aggregate streaming chunks into a final ChatMessage with reasoning content and thought signatures.

        This method extends the standard streaming chunk aggregation to handle Google GenAI's
        specific reasoning content, thinking token usage, and thought signatures for multi-turn context.

        :param chunks: List of streaming chunks to aggregate.
        :returns: Final ChatMessage with aggregated content, reasoning, and thought signatures.
        """

        # Use the generic aggregator for standard content (text, tool calls, basic meta)
        message = _convert_streaming_chunks_to_chat_message(chunks)

        # Now enhance with Google-specific features: reasoning content, thinking token usage, and thought signatures
        reasoning_text_parts: list[str] = []
        thought_signatures: List[Dict[str, Any]] = []
        thoughts_token_count = None

        for chunk in chunks:
            # Extract reasoning deltas
            if chunk.meta and "reasoning_deltas" in chunk.meta:
                reasoning_deltas = chunk.meta["reasoning_deltas"]
                if isinstance(reasoning_deltas, list):
                    for delta in reasoning_deltas:
                        if delta.get("type") == "reasoning":
                            reasoning_text_parts.append(delta.get("content", ""))

            # Extract thought signature deltas (for multi-turn context preservation)
            if chunk.meta and "thought_signature_deltas" in chunk.meta:
                signature_deltas = chunk.meta["thought_signature_deltas"]
                if isinstance(signature_deltas, list):
                    # Aggregate thought signatures - they should come from the final chunks
                    # We'll keep the last set of signatures as they represent the complete state
                    thought_signatures = signature_deltas

            # Extract thinking token usage (from the last chunk that has it)
            if chunk.meta and "usage" in chunk.meta:
                chunk_usage = chunk.meta["usage"]
                if "thoughts_token_count" in chunk_usage:
                    thoughts_token_count = chunk_usage["thoughts_token_count"]

        # Add thinking token count to usage if present
        if thoughts_token_count is not None and "usage" in message.meta:
            if message.meta["usage"] is None:
                message.meta["usage"] = {}
            message.meta["usage"]["thoughts_token_count"] = thoughts_token_count

        # Add thought signatures to meta if present (for multi-turn context preservation)
        if thought_signatures:
            message.meta["thought_signatures"] = thought_signatures

        # If we have reasoning content, reconstruct the message to include it
        # Note: ChatMessage doesn't support adding reasoning after creation, reconstruction is necessary
        if reasoning_text_parts:
            reasoning_content = ReasoningContent(reasoning_text="".join(reasoning_text_parts))
            return ChatMessage.from_assistant(
                text=message.text, tool_calls=message.tool_calls, meta=message.meta, reasoning=reasoning_content
            )

        return message

    def _handle_streaming_response(
        self, response_stream: Iterator[types.GenerateContentResponse], streaming_callback: StreamingCallbackT
    ) -> Dict[str, List[ChatMessage]]:
        """
        Handle streaming response from Google Gen AI generate_content_stream.
        :param response_stream: The streaming response from generate_content_stream.
        :param streaming_callback: The callback function for streaming chunks.
        :returns: A dictionary with the replies.
        """
        component_info = ComponentInfo.from_component(self)

        try:
            chunks = []

            chunk = None
            for i, chunk in enumerate(response_stream):
                streaming_chunk = self._convert_google_chunk_to_streaming_chunk(
                    chunk=chunk, index=i, component_info=component_info
                )
                chunks.append(streaming_chunk)

                # Stream the chunk
                streaming_callback(streaming_chunk)

            # Use custom aggregation that supports reasoning content
            message = self._aggregate_streaming_chunks_with_reasoning(chunks)
            return {"replies": [message]}

        except Exception as e:
            msg = f"Error in streaming response: {e}"
            raise RuntimeError(msg) from e

    async def _handle_streaming_response_async(
        self, response_stream: AsyncIterator[types.GenerateContentResponse], streaming_callback: AsyncStreamingCallbackT
    ) -> Dict[str, List[ChatMessage]]:
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

                streaming_chunk = self._convert_google_chunk_to_streaming_chunk(
                    chunk=chunk, index=i, component_info=component_info
                )
                chunks.append(streaming_chunk)

                # Stream the chunk
                await streaming_callback(streaming_chunk)

            # Use custom aggregation that supports reasoning content
            message = self._aggregate_streaming_chunks_with_reasoning(chunks)
            return {"replies": [message]}

        except Exception as e:
            msg = f"Error in async streaming response: {e}"
            raise RuntimeError(msg) from e

    def _process_thinking_config(self, generation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process thinking configuration from generation_kwargs.

        :param generation_kwargs: The generation configuration dictionary.
        :returns: Updated generation_kwargs with thinking_config if applicable.
        """
        if "thinking_budget" in generation_kwargs:
            thinking_budget = generation_kwargs.pop("thinking_budget")

            # Basic type validation
            if not isinstance(thinking_budget, int):
                logger.warning(
                    f"Invalid thinking_budget type: {type(thinking_budget)}. Expected int, using dynamic allocation."
                )
                thinking_budget = -1

            # Create thinking config
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=True)
            generation_kwargs["thinking_config"] = thinking_config

        return generation_kwargs

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ) -> Dict[str, Any]:
        """
        Run the Google Gen AI chat generator on the given input data.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Configuration for generation. If provided, it will override
        the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
        :param safety_settings: Safety settings for content filtering. If provided, it will override the
        default settings.
        :param streaming_callback: A callback function that is called when a new token is
        received from the stream.
        :param tools: A list of Tool objects or a Toolset that the model can use. If provided, it will
        override the tools set during initialization.
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
        generation_kwargs = self._process_thinking_config(generation_kwargs)

        # Select appropriate streaming callback
        streaming_callback = select_streaming_callback(
            init_callback=self._streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=False,
        )

        # Check for duplicate tool names
        _check_duplicate_tool_names(list(tools or []))  # handles Toolset as well

        # Handle system message if present
        system_instruction = None
        chat_messages = messages

        if messages and messages[0].is_from(ChatRole.SYSTEM):
            system_instruction = messages[0].text or ""
            chat_messages = messages[1:]

        # Convert messages to Google Gen AI Content format
        contents: List[types.ContentUnionDict] = []
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

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ) -> Dict[str, Any]:
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
        :param tools: A list of Tool objects or a Toolset that the model can use. If provided, it will
        override the tools set during initialization.
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
        generation_kwargs = self._process_thinking_config(generation_kwargs)

        # Select appropriate streaming callback
        streaming_callback = select_streaming_callback(
            init_callback=self._streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=True,
        )

        # Check for duplicate tool names
        _check_duplicate_tool_names(list(tools or []))  # handles Toolset as well

        # Handle system message if present
        system_instruction = None
        chat_messages = messages

        if messages and messages[0].is_from(ChatRole.SYSTEM):
            system_instruction = messages[0].text or ""
            chat_messages = messages[1:]

        # Convert messages to Google Gen AI Content format
        contents: List[types.ContentUnion] = []
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
