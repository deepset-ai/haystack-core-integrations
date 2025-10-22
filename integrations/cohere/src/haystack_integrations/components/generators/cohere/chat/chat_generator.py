import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union, get_args

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import ChatMessage, ComponentInfo, ImageContent, TextContent, ToolCall
from haystack.dataclasses.streaming_chunk import (
    AsyncStreamingCallbackT,
    FinishReason,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.tools import (
    Tool,
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from cohere import (
    AssistantChatMessageV2,
    AsyncClientV2,
    ChatResponse,
    ClientV2,
    ImageUrl,
    ImageUrlContent,
    StreamedChatResponseV2,
    SystemChatMessageV2,
    TextAssistantMessageV2ContentItem,
    TextSystemMessageV2ContentItem,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    Usage,
    UserChatMessageV2,
)
from cohere import (
    TextContent as CohereTextContent,
)

logger = logging.getLogger(__name__)


# Supported image formats based on Cohere's documentation
# See: https://docs.cohere.com/docs/image-inputs
ImageFormat = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
IMAGE_SUPPORTED_FORMATS: list[ImageFormat] = list(get_args(ImageFormat))


def _format_tool(tool: Tool) -> Dict[str, Any]:
    """
    Formats a Haystack Tool into Cohere's function specification format.

    The function transforms the tool's properties (name, description, parameters)
    into the structure expected by Cohere's API.

    :param tool: The Haystack Tool to format.
    :return: Dictionary formatted according to Cohere's function specification.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _format_message(
    message: ChatMessage,
) -> Union[UserChatMessageV2, AssistantChatMessageV2, SystemChatMessageV2, ToolChatMessageV2]:
    """
    Formats a Haystack ChatMessage into Cohere's chat format.

    The function handles message components including:
    - Text content
    - Image content (multimodal)
    - Tool calls
    - Tool call results

    :param message: Haystack ChatMessage to format.
    :return: A Cohere message object.
    """
    if not message.texts and not message.tool_calls and not message.tool_call_results and not message.images:
        msg = (
            "A `ChatMessage` must contain at least one `TextContent`, `ImageContent`, `ToolCall`, or `ToolCallResult`."
        )
        raise ValueError(msg)

    if message.images and not message.role.value == "user":
        msg = "`ImageContent` is only supported for user messages."
        raise ValueError(msg)

    # Format the message based on its content type
    if message.tool_call_results:
        result = message.tool_call_results[0]  # We expect one result at a time
        if result.origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Cohere."
            raise ValueError(msg)
        return ToolChatMessageV2(tool_call_id=result.origin.id, content=json.dumps({"result": result.result}))

    if message.tool_calls:
        tool_calls = []
        for tool_call in message.tool_calls:
            if tool_call.id is None:
                msg = "`ToolCall` must have a non-null `id` attribute to be used with Cohere."
                raise ValueError(msg)
            tool_calls.append(
                ToolCallV2(
                    id=tool_call.id,
                    type="function",
                    function=ToolCallV2Function(
                        name=tool_call.tool_name,
                        arguments=json.dumps(tool_call.arguments),
                    ),
                )
            )
        return AssistantChatMessageV2(
            tool_calls=tool_calls,
            tool_plan=message.text if message.text else "",
        )

    if message.role.value == "user":
        if not message.images and not message.text:
            msg = "A `ChatMessage` from user must contain at least one non-empty `TextContent` or `ImageContent`."
            raise ValueError(msg)

        # Handle multimodal content (text + images)
        if message.images:
            # Validate image formats
            for image in message.images:
                if image.mime_type not in IMAGE_SUPPORTED_FORMATS:
                    supported_formats = ", ".join(IMAGE_SUPPORTED_FORMATS)
                    msg = (
                        f"Unsupported image format: {image.mime_type}. "
                        f"Cohere supports the following formats: {supported_formats}"
                    )
                    raise ValueError(msg)

        # Build multimodal content following Cohere's API specification
        content_parts: List[Union[CohereTextContent, ImageUrlContent]] = []
        for part in message._content:
            if isinstance(part, TextContent) and part.text:
                text_content = CohereTextContent(text=part.text)
                content_parts.append(text_content)
            elif isinstance(part, ImageContent):
                # Cohere expects base64 data URI format
                # See: https://docs.cohere.com/docs/image-inputs
                image_url = f"data:{part.mime_type};base64,{part.base64_image}"
                image_content = ImageUrlContent(image_url=ImageUrl(url=image_url))
                content_parts.append(image_content)

        return UserChatMessageV2(content=content_parts)

    if message.role.value == "assistant":
        if not message.text:
            msg = "A `ChatMessage` from assistant without tool calls must contain a non-empty `TextContent`."
            raise ValueError(msg)
        return AssistantChatMessageV2(content=[TextAssistantMessageV2ContentItem(text=message.text)])

    if message.role.value == "system":
        if not message.text:
            msg = "A `ChatMessage` from system calls must contain a non-empty `TextContent`."
            raise ValueError(msg)
        return SystemChatMessageV2(content=[TextSystemMessageV2ContentItem(text=message.text)])

    msg = f"Unsupported message role: {message.role.value}"
    raise ValueError(msg)


def _parse_response(chat_response: ChatResponse, model: str) -> ChatMessage:
    """
    Parses Cohere's chat response into a Haystack ChatMessage.

    Extracts and organizes various response components including:
    - Text content
    - Tool calls
    - Usage statistics
    - Citations
    - Metadata

    :param chat_response: Response from Cohere's chat API.
    :param model: The name of the model that generated the response.
    :return: A Haystack ChatMessage containing the formatted response.
    """
    if chat_response.message.tool_calls:
        tool_calls = []
        for tc in chat_response.message.tool_calls:
            if tc.function and tc.function.name and tc.function.arguments and isinstance(tc.function.arguments, str):
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        tool_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        # Create message with tool plan as text and tool calls in the format Haystack expects
        tool_plan = chat_response.message.tool_plan or ""
        message = ChatMessage.from_assistant(text=tool_plan, tool_calls=tool_calls)
    elif chat_response.message.content and hasattr(chat_response.message.content[0], "text"):
        message = ChatMessage.from_assistant(chat_response.message.content[0].text)
    else:
        # Handle the case where neither tool_calls nor content exists
        logger.warning(f"Received empty response from Cohere API: {chat_response.message}")
        message = ChatMessage.from_assistant("")

    # In V2, token usage is part of the response object, not the message
    message._meta.update(
        {
            "model": model,
            "index": 0,
            "finish_reason": chat_response.finish_reason,
            "citations": chat_response.message.citations,
        }
    )
    if chat_response.usage and chat_response.usage.billed_units:
        message._meta["usage"] = {
            "prompt_tokens": chat_response.usage.billed_units.input_tokens,
            "completion_tokens": chat_response.usage.billed_units.output_tokens,
        }
    return message


def _convert_cohere_chunk_to_streaming_chunk(
    chunk: StreamedChatResponseV2,
    model: str,
    component_info: Optional[ComponentInfo] = None,
    global_index: int = 0,
) -> StreamingChunk:
    """
    Converts a Cohere streaming response chunk to a StreamingChunk.

    References the Cohere API documentation for the structure of the chunk.

    https://docs.cohere.com/reference/chat-stream
    https://docs.cohere.com/reference/chat#response.body.finish_reason

    :param chunk: The chunk returned by the Cohere API.
    :param component_info: An optional `ComponentInfo` object containing information about the component that
        generated the chunk, such as the component name and type.
    :param model: The model name for metadata.
    :param global_index: The index of the current block in the sequence of chunks.

    :returns:
        A StreamingChunk object representing the content of the chunk from the Cohere API.
    """
    finish_reason_mapping: Dict[str, FinishReason] = {
        "COMPLETE": "stop",
        "MAX_TOKENS": "length",
        "TOOL_CALLS": "tool_calls",
    }

    # Initialize default values
    content = ""
    index = global_index
    start = False
    finish_reason = None
    tool_calls = None
    meta: Dict[str, Any] = {"model": model}

    if chunk.type == "content-delta" and chunk.delta and chunk.delta.message:
        if chunk.delta.message and chunk.delta.message.content and chunk.delta.message.content.text is not None:
            content = chunk.delta.message.content.text

    elif chunk.type == "tool-plan-delta" and chunk.delta and chunk.delta.message:
        if chunk.delta.message and chunk.delta.message.tool_plan is not None:
            content = chunk.delta.message.tool_plan

    elif chunk.type == "tool-call-start" and chunk.delta and chunk.delta.message:
        if chunk.delta.message and chunk.delta.message.tool_calls:
            tool_call = chunk.delta.message.tool_calls
            function = tool_call.function
            if function is not None and function.name is not None:
                tool_calls = [
                    ToolCallDelta(
                        index=global_index,
                        id=tool_call.id,
                        tool_name=function.name,
                        arguments=None,
                    )
                ]
                start = True  # This starts a tool call
                if tool_call.id is not None:
                    meta["tool_call_id"] = tool_call.id

    elif chunk.type == "tool-call-delta" and chunk.delta and chunk.delta.message:
        if (
            chunk.delta.message
            and chunk.delta.message.tool_calls
            and chunk.delta.message.tool_calls.function
            and chunk.delta.message.tool_calls.function.arguments is not None
        ):
            arguments = chunk.delta.message.tool_calls.function.arguments
            tool_calls = [
                ToolCallDelta(
                    index=global_index,
                    tool_name=None,
                    arguments=arguments,
                )
            ]

    elif chunk.type == "tool-call-end":
        # Tool call end doesn't have content, just signals completion
        start = True

    elif chunk.type == "message-end":
        finish_reason_raw = getattr(chunk.delta, "finish_reason", None)
        finish_reason = finish_reason_mapping.get(finish_reason_raw) if finish_reason_raw else None

        # The Cohere API is subject to changes in how usage data is returned. We try to support both dict and objects.
        usage_data = getattr(chunk.delta, "usage", None)
        prompt_tokens, completion_tokens = 0.0, 0.0
        if isinstance(usage_data, dict):
            try:
                prompt_tokens = usage_data["billed_units"]["input_tokens"]
                completion_tokens = usage_data["billed_units"]["output_tokens"]
            except KeyError:
                pass
        elif (
            usage_data is not None
            and isinstance(usage_data, Usage)
            and usage_data.billed_units
            and usage_data.billed_units.input_tokens is not None
            and usage_data.billed_units.output_tokens is not None
        ):
            prompt_tokens = usage_data.billed_units.input_tokens
            completion_tokens = usage_data.billed_units.output_tokens

        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        meta["finish_reason"] = finish_reason_raw
        meta["usage"] = usage

    return StreamingChunk(
        content=content,
        component_info=component_info,
        index=index,
        tool_calls=tool_calls,
        start=start,
        finish_reason=finish_reason,
        meta=meta,
    )


def _parse_streaming_response(
    response: Iterator[StreamedChatResponseV2],
    model: str,
    streaming_callback: SyncStreamingCallbackT,
    component_info: ComponentInfo,
) -> ChatMessage:
    """
    Parses Cohere's streaming chat response.

    Loops through each stream object from Cohere and converts it into a StreamingChunk.
    """
    chunks: List[StreamingChunk] = []
    global_index = 0

    for chunk in response:
        if chunk.type in ["tool-call-start", "content-start", "citation-start"]:
            global_index += 1

        streaming_chunk = _convert_cohere_chunk_to_streaming_chunk(
            chunk=chunk,
            component_info=component_info,
            model=model,
            global_index=global_index,
        )

        if not streaming_chunk:
            continue

        chunks.append(streaming_chunk)
        streaming_callback(streaming_chunk)

    return _convert_streaming_chunks_to_chat_message(chunks=chunks)


async def _parse_async_streaming_response(
    response: AsyncIterator[StreamedChatResponseV2],
    model: str,
    streaming_callback: AsyncStreamingCallbackT,
    component_info: ComponentInfo,
) -> ChatMessage:
    """
    Parses Cohere's async streaming chat response into a Haystack ChatMessage.
    """
    chunks: List[StreamingChunk] = []
    global_index = 0

    async for chunk in response:
        if chunk.type in ["tool-call-start", "content-start", "citation-start"]:
            global_index += 1

        streaming_chunk = _convert_cohere_chunk_to_streaming_chunk(
            chunk=chunk, component_info=component_info, model=model, global_index=global_index
        )
        if not streaming_chunk:
            continue

        chunks.append(streaming_chunk)
        await streaming_callback(streaming_chunk)

    return _convert_streaming_chunks_to_chat_message(chunks=chunks)


@component
class CohereChatGenerator:
    """
    Completes chats using Cohere's models using cohere.ClientV2 `chat` endpoint.

    This component supports both text-only and multimodal (text + image) conversations
    using Cohere's vision models like Command A Vision.

    Supported image formats: PNG, JPEG, WEBP, GIF (non-animated).
    Maximum 20 images per request with 20MB total limit.

    You can customize how the chat response is generated by passing parameters to the
    Cohere API through the `**generation_kwargs` parameter. You can do this when
    initializing or running the component. Any parameter that works with
     `cohere.ClientV2.chat` will work here too.
    For details, see [Cohere API](https://docs.cohere.com/reference/chat).

    Below is an example of how to use the component:

    ### Simple example
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret
    from haystack_integrations.components.generators.cohere import CohereChatGenerator

    client = CohereChatGenerator(model="command-r-08-2024", api_key=Secret.from_env_var("COHERE_API_KEY"))
    messages = [ChatMessage.from_user("What's Natural Language Processing?")]
    client.run(messages)

    # Output: {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>,
    # _content=[TextContent(text='Natural Language Processing (NLP) is an interdisciplinary...
    ```

    ### Multimodal example
    ```python
    from haystack.dataclasses import ChatMessage, ImageContent
    from haystack.utils import Secret
    from haystack_integrations.components.generators.cohere import CohereChatGenerator

    # Create an image from file path or base64
    image_content = ImageContent.from_file_path("path/to/your/image.jpg")

    # Create a multimodal message with both text and image
    messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

    # Use a multimodal model like Command A Vision
    client = CohereChatGenerator(model="command-a-vision-07-2025", api_key=Secret.from_env_var("COHERE_API_KEY"))
    response = client.run(messages)
    print(response)
    ```

    ### Advanced example

    CohereChatGenerator can be integrated into pipelines and supports Haystack's tooling
    architecture, enabling tools to be invoked seamlessly across various generators.

    ```python
    from haystack import Pipeline
    from haystack.dataclasses import ChatMessage
    from haystack.components.tools import ToolInvoker
    from haystack.tools import Tool
    from haystack_integrations.components.generators.cohere import CohereChatGenerator

    # Create a weather tool
    def weather(city: str) -> str:
        return f"The weather in {city} is sunny and 32°C"

    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for, e.g. Paris, London",
                }
            },
            "required": ["city"],
        },
        function=weather,
    )

    # Create and set up the pipeline
    pipeline = Pipeline()
    pipeline.add_component("generator", CohereChatGenerator(model="command-r-08-2024", tools=[weather_tool]))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=[weather_tool]))
    pipeline.connect("generator", "tool_invoker")

    # Run the pipeline with a weather query
    results = pipeline.run(
        data={"generator": {"messages": [ChatMessage.from_user("What's the weather like in Paris?")]}}
    )

    # The tool result will be available in the pipeline output
    print(results["tool_invoker"]["tool_messages"][0].tool_call_result.result)
    # Output: "The weather in Paris is sunny and 32°C"
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "command-r-08-2024",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        **kwargs: Any,
    ):
        """
        Initialize the CohereChatGenerator instance.

        :param api_key: The API key for the Cohere API.
        :param model: The name of the model to use. You can use models from the `command` family.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
            as an argument.
        :param api_base_url: The base URL of the Cohere API.
        :param generation_kwargs: Other parameters to use for the model during generation. For a list of parameters,
            see [Cohere Chat endpoint](https://docs.cohere.com/reference/chat).
            Some of the parameters are:
            - 'messages': A list of messages between the user and the model, meant to give the model
              conversational context for responding to the user's message.
            - 'system_message': When specified, adds a system message at the beginning of the conversation.
            - 'citation_quality': Defaults to `accurate`. Dictates the approach taken to generating citations
              as part of the RAG flow by allowing the user to specify whether they want
              `accurate` results or `fast` results.
            - 'temperature': A non-negative float that tunes the degree of randomness in generation. Lower temperatures
              mean less random generations.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset that the model can use.
            Each tool should have a unique name.

        """
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        if not api_base_url:
            api_base_url = "https://api.cohere.com"
        if generation_kwargs is None:
            generation_kwargs = {}
        self.api_key = api_key
        self.model = model
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs
        self.tools = tools
        self.model_parameters = kwargs
        self.client = ClientV2(
            api_key=self.api_key.resolve_value(),
            base_url=self.api_base_url,
            client_name="haystack",
        )
        self.async_client = AsyncClientV2(
            api_key=self.api_key.resolve_value(),
            base_url=self.api_base_url,
            client_name="haystack",
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
                Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            api_key=self.api_key.to_dict(),
            generation_kwargs=self.generation_kwargs,
            tools=serialize_tools_or_toolset(self.tools),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
               Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        deserialize_tools_or_toolset_inplace(init_params, key="tools")
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Invoke the chat endpoint based on the provided messages and generation parameters.

        :param messages: list of `ChatMessage` instances representing the input messages.
        :param generation_kwargs: additional keyword arguments for chat generation. These parameters will
            potentially override the parameters passed in the __init__ method.
            For more details on the parameters supported by the Cohere API, refer to the
            Cohere [documentation](https://docs.cohere.com/reference/chat).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter set during component initialization.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.

        :returns: A dictionary with the following keys:
            - `replies`: a list of `ChatMessage` instances representing the generated responses.
        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {
            **self.generation_kwargs,
            **(generation_kwargs or {}),
        }

        # Handle tools
        tools = tools or self.tools
        flattened_tools = flatten_tools_or_toolsets(tools)
        if flattened_tools:
            _check_duplicate_tool_names(flattened_tools)
            generation_kwargs["tools"] = [_format_tool(tool) for tool in flattened_tools]

        formatted_messages = [_format_message(message) for message in messages]

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        if streaming_callback:
            component_info = ComponentInfo.from_component(self)
            streamed_response = self.client.chat_stream(
                model=self.model,
                messages=formatted_messages,
                **generation_kwargs,
            )
            chat_message = _parse_streaming_response(
                response=streamed_response,
                model=self.model,
                streaming_callback=streaming_callback,
                component_info=component_info,
            )
        else:
            response = self.client.chat(
                model=self.model,
                messages=formatted_messages,
                **generation_kwargs,
            )
            chat_message = _parse_response(response, self.model)

        return {"replies": [chat_message]}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Asynchronously invoke the chat endpoint based on the provided messages and generation parameters.

        :param messages: list of `ChatMessage` instances representing the input messages.
        :param generation_kwargs: additional keyword arguments for chat generation. These parameters will
            potentially override the parameters passed in the __init__ method.
            For more details on the parameters supported by the Cohere API, refer to the
            Cohere [documentation](https://docs.cohere.com/reference/chat).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter set during component initialization.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :returns: A dictionary with the following keys:
            - `replies`: a list of `ChatMessage` instances representing the generated responses.
        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {
            **self.generation_kwargs,
            **(generation_kwargs or {}),
        }

        # Handle tools
        tools = tools or self.tools
        flattened_tools = flatten_tools_or_toolsets(tools)
        if flattened_tools:
            _check_duplicate_tool_names(flattened_tools)
            generation_kwargs["tools"] = [_format_tool(tool) for tool in flattened_tools]

        formatted_messages = [_format_message(message) for message in messages]

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        if streaming_callback:
            component_info = ComponentInfo.from_component(self)
            streamed_response = self.async_client.chat_stream(
                model=self.model,
                messages=formatted_messages,
                **generation_kwargs,
            )
            chat_message = await _parse_async_streaming_response(
                response=streamed_response,
                model=self.model,
                streaming_callback=streaming_callback,
                component_info=component_info,
            )
        else:
            response = await self.async_client.chat(
                model=self.model,
                messages=formatted_messages,
                **generation_kwargs,
            )
            chat_message = _parse_response(response, self.model)

        return {"replies": [chat_message]}
