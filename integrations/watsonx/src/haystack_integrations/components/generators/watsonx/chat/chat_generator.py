# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal, get_args

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FinishReason,
    ImageContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    TextContent,
    ToolCall,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

logger = logging.getLogger(__name__)


# See https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-prompt-data.html?context=wx
# for supported formats
ImageFormat = Literal["image/jpeg", "image/png"]
IMAGE_SUPPORTED_FORMATS: list[ImageFormat] = list(get_args(ImageFormat))

# See https://ibm.github.io/watsonx-ai-node-sdk/enums/1_6_x.WatsonXAI.TextChatResultChoiceStream.Constants.FinishReason.html
# for possible finish reasons
FINISH_REASON_MAPPING: dict[str, FinishReason] = {
    "cancelled": "stop",
    "error": "stop",
    "length": "length",
    "stop": "stop",
    "time_limit": "stop",
    "tool_calls": "tool_calls",
}


@component
class WatsonxChatGenerator:
    """
    Enables chat completions using IBM's watsonx.ai foundation models.

    This component interacts with IBM's watsonx.ai platform to generate chat responses using various foundation
    models. It supports the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format for both input
    and output, including multimodal inputs with text and images.

    The generator works with IBM's foundation models that are listed
    [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&audience=wdp).

    You can customize the generation behavior by passing parameters to the watsonx.ai API through the
    `generation_kwargs` argument. These parameters are passed directly to the watsonx.ai inference endpoint.

    For details on watsonx.ai API parameters, see
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    messages = [ChatMessage.from_user("Explain quantum computing in simple terms")]

    client = WatsonxChatGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        model="ibm/granite-4-h-small",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )
    response = client.run(messages)
    print(response)
    ```

    ### Multimodal usage example

    ```python
    from haystack.dataclasses import ChatMessage, ImageContent

    # Create an image from file path or base64
    image_content = ImageContent.from_file_path("path/to/your/image.jpg")

    # Create a multimodal message with both text and image
    messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

    # Use a multimodal model
    client = WatsonxChatGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        model="meta-llama/llama-3-2-11b-vision-instruct",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )
    response = client.run(messages)
    print(response)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "ibm/granite-3-1-8b-base",
        "ibm/granite-3-8b-instruct",
        "ibm/granite-4-h-small",
        "ibm/granite-8b-code-instruct",
        "ibm/granite-guardian-3-8b",
        "meta-llama/llama-3-1-70b-gptq",
        "meta-llama/llama-3-1-8b",
        "meta-llama/llama-3-2-11b-vision-instruct",
        "meta-llama/llama-3-2-90b-vision-instruct",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-3-405b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "meta-llama/llama-guard-3-11b-vision",
        "mistral-large-2512",
        "mistralai/mistral-medium-2505",
        "mistralai/mistral-small-3-1-24b-instruct-2503",
        "openai/gpt-oss-120b",
    ]
    """A non-exhaustive list of models supported by this component.

    See https://www.ibm.com/docs/en/watsonx/saas?topic=solutions-supported-foundation-models for the
    full list of models and up-to-date model IDs.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),  # noqa: B008
        model: str = "ibm/granite-4-h-small",
        project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),  # noqa: B008
        api_base_url: str = "https://us-south.ml.cloud.ibm.com",
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        verify: bool | str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> None:
        """
        Creates an instance of WatsonxChatGenerator.

        Before initializing the component, you can set environment variables:
        - `WATSONX_TIMEOUT` to override the default timeout
        - `WATSONX_MAX_RETRIES` to override the default retry count

        :param api_key: IBM Cloud API key for watsonx.ai access.
            Can be set via `WATSONX_API_KEY` environment variable or passed directly.
        :param model: The model ID to use for completions. Defaults to "ibm/granite-4-h-small".
            Available models can be found in your IBM Cloud account.
        :param project_id: IBM Cloud project ID
        :param api_base_url: Custom base URL for the API endpoint.
            Defaults to "https://us-south.ml.cloud.ibm.com".
        :param generation_kwargs: Additional parameters to control text generation.
            These parameters are passed directly to the watsonx.ai inference endpoint.
            Supported parameters include:
            - `temperature`: Controls randomness (lower = more deterministic)
            - `max_new_tokens`: Maximum number of tokens to generate
            - `min_new_tokens`: Minimum number of tokens to generate
            - `top_p`: Nucleus sampling probability threshold
            - `top_k`: Number of highest probability tokens to consider
            - `repetition_penalty`: Penalty for repeated tokens
            - `length_penalty`: Penalty based on output length
            - `stop_sequences`: List of sequences where generation should stop
            - `random_seed`: Seed for reproducible results
        :param timeout: Timeout in seconds for API requests.
            Defaults to environment variable `WATSONX_TIMEOUT` or 30 seconds.
        :param max_retries: Maximum number of retry attempts for failed requests.
            Defaults to environment variable `WATSONX_MAX_RETRIES` or 5.
        :param verify: SSL verification setting. Can be:
            - True: Verify SSL certificates (default)
            - False: Skip verification (insecure)
            - Path to CA bundle for custom certificates
        :param streaming_callback: A callback function for streaming responses.
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
        """
        self.api_key = api_key
        self.model = model
        self.project_id = project_id
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify = verify
        self.streaming_callback = streaming_callback
        self.tools = tools

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Watsonx client with configured credentials."""
        credentials = Credentials(api_key=self.api_key.resolve_value(), url=self.api_base_url)

        self.client = ModelInference(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id.resolve_value(),
            verify=self.verify,
            max_retries=self.max_retries,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        serialized_tools = serialize_tools_or_toolset(self.tools) if self.tools else None
        return default_to_dict(
            self,
            model=self.model,
            project_id=self.project_id.to_dict(),
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            verify=self.verify,
            streaming_callback=callback_name,
            tools=serialized_tools,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatsonxChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "project_id"])
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback = init_params.get("streaming_callback")
        if serialized_callback:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback)
        return default_from_dict(cls, data)

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        *,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Generate chat completions synchronously.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided this will override the `streaming_callback` set in the `__init__` method.
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if not messages:
            return {"replies": []}

        resolved_streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        api_args = self._prepare_api_call(messages=messages, generation_kwargs=generation_kwargs, tools=tools)

        if resolved_streaming_callback:
            return self._handle_streaming(api_args=api_args, callback=resolved_streaming_callback)

        return self._handle_standard(api_args)

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        *,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Generate chat completions asynchronously.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided this will override the `streaming_callback` set in the `__init__` method.
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if not messages:
            return {"replies": []}

        resolved_streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        api_args = self._prepare_api_call(messages=messages, generation_kwargs=generation_kwargs, tools=tools)

        if resolved_streaming_callback:
            return await self._handle_async_streaming(api_args=api_args, callback=resolved_streaming_callback)

        return await self._handle_async_standard(api_args)

    def _prepare_api_call(
        self,
        *,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, Any]:
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        watsonx_messages = []
        content: str | None | dict[str, Any] | list[dict[str, Any]]

        flattened_tools = flatten_tools_or_toolsets(tools or self.tools)
        _check_duplicate_tool_names(flattened_tools)
        tool_definitions = [{"type": "function", "function": {**tool.tool_spec}} for tool in flattened_tools]

        for msg in messages:
            # Watsonx tool call result messages are of the same format as OpenAI chat completions
            if msg.tool_call_results:
                watsonx_messages.append(msg.to_openai_dict_format(require_tool_call_ids=True))
                continue

            # Check that images are only in user messages
            if msg.images and not msg.is_from(ChatRole.USER):
                error_msg = "Image content is only supported for user messages"
                raise ValueError(error_msg)

            # Handle multimodal content (text + images) preserving order
            if msg.images:
                # Pre-validate all images first (following LlamaCpp pattern)
                for image in msg.images:
                    if image.mime_type not in IMAGE_SUPPORTED_FORMATS:
                        supported_formats = ", ".join(IMAGE_SUPPORTED_FORMATS)
                        msg_error = (
                            f"Unsupported image format: {image.mime_type}. "
                            f"WatsonX supports the following formats: {supported_formats}"
                        )
                        raise ValueError(msg_error)

                content_parts: list[dict[str, Any]] = []
                for part in msg._content:
                    if isinstance(part, TextContent) and part.text:
                        content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        # WatsonX expects base64 data URI format
                        # See: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-api-chat.html?context=wx
                        image_url = f"data:{part.mime_type};base64,{part.base64_image}"
                        content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

                content = content_parts
            else:
                # Simple text-only message
                content = msg.text

            watsonx_msg = {"role": msg.role.value, "content": content}
            if msg.name:
                watsonx_msg["name"] = msg.name
            watsonx_messages.append(watsonx_msg)

        merged_kwargs.pop("stream", None)

        api_args = {"messages": watsonx_messages, "params": merged_kwargs}
        if tool_definitions:
            api_args["tools"] = tool_definitions

        return api_args

    def _convert_chunk_to_streaming_chunk(self, chunk: dict[str, Any], component_info: ComponentInfo) -> StreamingChunk:
        """
        Convert one Watsonx AI stream-chunk to Haystack StreamingChunk.
        """
        choice = chunk["choices"][0]
        chunk_meta = {
            "model": self.model,
            "model_id": chunk.get("model_id"),
            "model_version": chunk.get("model_version"),
            "created": chunk.get("created"),
            "created_at": chunk.get("created_at"),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

        if choice["delta"] and (choice_delta_tool_calls := choice["delta"].get("tool_calls")):
            # create a list of ToolCallDelta objects from the tool calls
            tool_calls_deltas = [
                ToolCallDelta(
                    index=tool_call["index"],
                    id=tool_call.get("id"),
                    tool_name=tool_call.get("function", {}).get("name"),
                    arguments=tool_call.get("function", {}).get("arguments"),
                )
                for tool_call in choice_delta_tool_calls
            ]
            return StreamingChunk(
                content=choice.get("delta", {}).get("content", ""),
                meta=chunk_meta,
                component_info=component_info,
                # We adopt the first tool_calls_deltas.index as the overall index of the chunk to match OpenAI
                index=tool_calls_deltas[0].index,
                tool_calls=tool_calls_deltas,
                start=tool_calls_deltas[0].tool_name is not None,
                finish_reason=FINISH_REASON_MAPPING.get(choice.get("finish_reason")),
            )

        index = choice.get("index", 0)
        return StreamingChunk(
            content=choice.get("delta", {}).get("content", ""),
            meta=chunk_meta,
            component_info=component_info,
            index=index,
            start=index == 0,
            finish_reason=FINISH_REASON_MAPPING.get(choice.get("finish_reason")),
        )

    def _handle_streaming(
        self,
        *,
        api_args: dict[str, Any],
        callback: SyncStreamingCallbackT,
    ) -> dict[str, list[ChatMessage]]:
        """
        Handle synchronous streaming response.

        :param api_args: Arguments for the API call, including messages and parameters.
        :param callback: A callback function to handle streaming chunks.
        :returns:
            A dictionary with the generated responses as ChatMessage instances.
        """
        chunks: list[StreamingChunk] = []
        stream = self.client.chat_stream(
            messages=api_args["messages"], params=api_args["params"], tools=api_args.get("tools")
        )
        component_info = ComponentInfo.from_component(self)

        for chunk in stream:
            if not isinstance(chunk, dict) or not chunk.get("choices"):
                continue

            streaming_chunk = self._convert_chunk_to_streaming_chunk(chunk, component_info)
            chunks.append(streaming_chunk)
            callback(streaming_chunk)

        chat_message = _convert_streaming_chunks_to_chat_message(chunks)
        message_tool_calls = [
            replace(tool_call, arguments=self._parse_tool_call_json(tool_call.arguments))
            for tool_call in chat_message.tool_calls
        ]
        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=chat_message.text,
                    meta=chat_message.meta,
                    tool_calls=message_tool_calls,
                    reasoning=chat_message.reasoning,
                )
            ]
        }

    def _handle_standard(self, api_args: dict[str, Any]) -> dict[str, list[ChatMessage]]:
        """Handle synchronous standard response."""
        response = self.client.chat(
            messages=api_args["messages"], params=api_args["params"], tools=api_args.get("tools")
        )
        return self._process_response(response)

    async def _handle_async_streaming(
        self,
        *,
        api_args: dict[str, Any],
        callback: AsyncStreamingCallbackT,
    ) -> dict[str, list[ChatMessage]]:
        """Handle asynchronous streaming response."""
        chunks: list[StreamingChunk] = []
        stream_generator = await self.client.achat_stream(
            messages=api_args["messages"], params=api_args["params"], tools=api_args.get("tools")
        )
        component_info = ComponentInfo.from_component(self)

        async for chunk in stream_generator:
            if not isinstance(chunk, dict) or not chunk.get("choices"):
                continue

            streaming_chunk = self._convert_chunk_to_streaming_chunk(chunk, component_info)
            chunks.append(streaming_chunk)
            await callback(streaming_chunk)

        chat_message = _convert_streaming_chunks_to_chat_message(chunks)
        message_tool_calls = [
            replace(tool_call, arguments=self._parse_tool_call_json(tool_call.arguments))
            for tool_call in chat_message.tool_calls
        ]
        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=chat_message.text,
                    meta=chat_message.meta,
                    tool_calls=message_tool_calls,
                    reasoning=chat_message.reasoning,
                )
            ]
        }

    async def _handle_async_standard(self, api_args: dict[str, Any]) -> dict[str, list[ChatMessage]]:
        """Handle asynchronous standard response."""
        response = await self.client.achat(
            messages=api_args["messages"], params=api_args["params"], tools=api_args.get("tools")
        )
        return self._process_response(response)

    @staticmethod
    def _parse_tool_call_json(tool_call: str | dict) -> dict[str, Any]:
        """Parse tool call json from Watsonx tool calls."""
        if isinstance(tool_call, dict):
            return tool_call
        obj = json.loads(tool_call)
        if isinstance(obj, str):
            obj = json.loads(obj)
        return obj

    def _process_response(self, response: dict[str, Any]) -> dict[str, list[ChatMessage]]:
        """Process standard response into Haystack format."""
        if not response.get("choices"):
            return {"replies": []}

        choices = response["choices"]
        chat_messages = []
        for choice in choices:
            message = choice.get("message", {})

            message_tool_calls: list[ToolCall] | None = None
            if tool_calls := message.get("tool_calls", []):
                message_tool_calls = [
                    ToolCall(
                        id=tool_call["id"],
                        tool_name=tool_call["function"]["name"],
                        arguments=self._parse_tool_call_json(tool_call["function"]["arguments"]),
                    )
                    for tool_call in tool_calls
                ]

            chat_messages.append(
                ChatMessage.from_assistant(
                    text=message.get("content", ""),
                    tool_calls=message_tool_calls,
                    meta={
                        "model": self.model,
                        "index": choice.get("index", 0),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": response.get("usage", {}),
                    },
                )
            )

        return {"replies": chat_messages}
