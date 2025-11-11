import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    ImageContent,
    StreamingCallbackT,
    TextContent,
    ToolCall,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.dataclasses.streaming_chunk import FinishReason, StreamingChunk, SyncStreamingCallbackT
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import deserialize_callable, serialize_callable
from llama_cpp import (
    ChatCompletionMessageToolCall,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPart,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionStreamResponseDeltaEmpty,
    ChatCompletionTool,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
    llama_chat_format,
)
from llama_cpp.llama_chat_format import Llava15ChatHandler
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

logger = logging.getLogger(__name__)

FINISH_REASON_MAPPING: Dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
}

SUPPORTED_IMAGE_FORMATS = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]


def _convert_message_to_llamacpp_format(message: ChatMessage) -> ChatCompletionRequestMessage:
    """
    Convert a ChatMessage to the format expected by llama.cpp Chat API.
    """
    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results
    images = message.images

    if not text_contents and not tool_calls and not tool_call_results and not images:
        msg = (
            "A `ChatMessage` must contain at least one `TextContent`, `ImageContent`, `ToolCall`, or `ToolCallResult`."
        )
        raise ValueError(msg)
    elif len(text_contents) + len(tool_call_results) > 1:
        msg = "For llama.cpp compatibility, a `ChatMessage` can contain at most one `TextContent` or `ToolCallResult`."
        raise ValueError(msg)

    role = message._role.value

    # Check that images are only in user messages
    if images and role != "user":
        msg = "Image content is only supported for user messages"
        raise ValueError(msg)

    if role == "tool" and tool_call_results:
        if tool_call_results[0].origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with llama.cpp."
            raise ValueError(msg)
        return {
            "role": "function",
            "content": tool_call_results[0].result,
            "name": tool_call_results[0].origin.tool_name,
        }

    if role == "system":
        return {"role": "system", "content": text_contents[0]}

    if role == "user":
        # Handle multimodal content (text + images) preserving order
        if images:
            # Check image constraints for LlamaCpp
            for image in images:
                if image.mime_type not in SUPPORTED_IMAGE_FORMATS:
                    supported_formats = ", ".join(SUPPORTED_IMAGE_FORMATS)
                    msg = (
                        f"Unsupported image format: {image.mime_type}. "
                        f"LlamaCpp supports the following formats: {supported_formats}"
                    )
                    raise ValueError(msg)

            content_parts: list[ChatCompletionRequestMessageContentPart] = []
            for part in message._content:
                if isinstance(part, TextContent) and part.text:
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    # LlamaCpp expects base64 data URI format
                    image_url = f"data:{part.mime_type};base64,{part.base64_image}"
                    content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

            return {"role": "user", "content": content_parts}

        # Simple text-only message
        return {"role": "user", "content": text_contents[0]}

    if role == "assistant":
        result: ChatCompletionRequestAssistantMessage = {"role": "assistant"}

        if text_contents:
            result["content"] = text_contents[0]

        if tool_calls:
            llamacpp_tool_calls: List[ChatCompletionMessageToolCall] = []
            for tc in tool_calls:
                if tc.id is None:
                    msg = "`ToolCall` must have a non-null `id` attribute to be used with llama.cpp."
                    raise ValueError(msg)
                llamacpp_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        # We disable ensure_ascii so special chars like emojis are not converted
                        "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                )
            result["tool_calls"] = llamacpp_tool_calls

        return result

    error_msg = f"Unknown role: {role}"
    raise ValueError(error_msg)


@component
class LlamaCppChatGenerator:
    """
    Provides an interface to generate text using LLM via llama.cpp.

    [llama.cpp](https://github.com/ggml-org/llama.cpp) is a project written in C/C++ for efficient inference of LLMs.
    It employs the quantized GGUF format, suitable for running these models on standard machines (even without GPUs).
    Supports both text-only and multimodal (text + image) models like LLaVA.

    Usage example:
    ```python
    from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator
    user_message = [ChatMessage.from_user("Who is the best American actor?")]
    generator = LlamaCppGenerator(model="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)

    print(generator.run(user_message, generation_kwargs={"max_tokens": 128}))
    # {"replies": [ChatMessage(content="John Cusack", role=<ChatRole.ASSISTANT: "assistant">, name=None, meta={...})}
    ```

    Usage example with multimodal (image + text):
    ```python
    from haystack.dataclasses import ChatMessage, ImageContent

    # Create an image from file path or base64
    image_content = ImageContent.from_file_path("path/to/your/image.jpg")

    # Create a multimodal message with both text and image
    messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

    # Initialize with multimodal support
    generator = LlamaCppChatGenerator(
        model="llava-v1.5-7b-q4_0.gguf",
        chat_handler_name="Llava15ChatHandler",  # Use llava-1-5 handler
        model_clip_path="mmproj-model-f16.gguf",  # CLIP model
        n_ctx=4096  # Larger context for image processing
    )
    generator.warm_up()

    result = generator.run(messages)
    print(result)
    ```
    """

    def __init__(
        self,
        model: str,
        n_ctx: Optional[int] = 0,
        n_batch: Optional[int] = 512,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        chat_handler_name: Optional[str] = None,
        model_clip_path: Optional[str] = None,
    ):
        """
        :param model: The path of a quantized model for text generation, for example, "zephyr-7b-beta.Q4_0.gguf".
            If the model path is also specified in the `model_kwargs`, this parameter will be ignored.
        :param n_ctx: The number of tokens in the context. When set to 0, the context will be taken from the model.
        :param n_batch: Prompt processing maximum batch size.
        :param model_kwargs: Dictionary containing keyword arguments used to initialize the LLM for text generation.
            These keyword arguments provide fine-grained control over the model loading.
            In case of duplication, these kwargs override `model`, `n_ctx`, and `n_batch` init parameters.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__).
        :param generation_kwargs:  A dictionary containing keyword arguments to customize text generation.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param chat_handler_name: Name of the chat handler for multimodal models.
            Common options include: "Llava16ChatHandler", "MoondreamChatHandler", "Qwen25VLChatHandler".
            For other handlers, check
            [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/#multi-modal-models).
        :param model_clip_path: Path to the CLIP model for vision processing (e.g., "mmproj.bin").
            Required when chat_handler_name is provided for multimodal models.
        """

        model_kwargs = model_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        # check if the model_kwargs contain the essential parameters
        # otherwise, populate them with values from init parameters
        model_kwargs.setdefault("model_path", model)
        model_kwargs.setdefault("n_ctx", n_ctx)
        model_kwargs.setdefault("n_batch", n_batch)

        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))

        handler: Optional[Llava15ChatHandler] = None
        # Validate multimodal requirements
        if chat_handler_name is not None:
            if model_clip_path is None:
                msg = "model_clip_path must be provided when chat_handler_name is specified for multimodal models"
                raise ValueError(msg)
            # Validate chat handler by attempting to import it
            try:
                handler = getattr(llama_chat_format, chat_handler_name)
            except AttributeError as e:
                msg = f"Failed to import chat handler '{chat_handler_name}'."
                raise ValueError(msg) from e

        self.model_path = model
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self._model: Optional[Llama] = None
        self.tools = tools
        self.streaming_callback = streaming_callback
        self.chat_handler_name = chat_handler_name
        self.model_clip_path = model_clip_path
        self._handler = handler

    def warm_up(self):
        if self._model is not None:
            return

        kwargs = self.model_kwargs.copy()
        if "hf_tokenizer_path" in kwargs and "tokenizer" not in kwargs:
            tokenizer = LlamaHFTokenizer.from_pretrained(kwargs["hf_tokenizer_path"])
            kwargs["tokenizer"] = tokenizer

        # Handle multimodal initialization
        if self._handler is not None and self.model_clip_path is not None:
            # the following command is correct, but mypy complains because handlers also have a __call__ method
            kwargs["chat_handler"] = self._handler(clip_model_path=self.model_clip_path)  # type: ignore[call-arg]

        self._model = Llama(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
              Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            model_kwargs=self.model_kwargs,
            generation_kwargs=self.generation_kwargs,
            tools=serialize_tools_or_toolset(self.tools),
            streaming_callback=callback_name,
            chat_handler_name=self.chat_handler_name,
            model_clip_path=self.model_clip_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaCppChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        if (
            "streaming_callback" in data["init_parameters"]
            and data["init_parameters"]["streaming_callback"] is not None
        ):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(
                data["init_parameters"]["streaming_callback"]
            )
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Run the text generation model on the given list of ChatMessages.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:  A dictionary containing keyword arguments to customize text generation.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name. If set, it will override the `tools` parameter set during
            component initialization.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            If set, it will override the `streaming_callback` parameter set during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        if self._model is None:
            error_msg = "The model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(error_msg)

        if not messages:
            return {"replies": []}

        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        formatted_messages = [_convert_message_to_llamacpp_format(msg) for msg in messages]

        tools = tools or self.tools
        flattened_tools = flatten_tools_or_toolsets(tools)
        _check_duplicate_tool_names(flattened_tools)

        llamacpp_tools: List[ChatCompletionTool] = []
        if flattened_tools:
            for t in flattened_tools:
                llamacpp_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.tool_spec["name"],
                            "description": t.tool_spec.get("description", ""),
                            "parameters": t.tool_spec.get("parameters", {}),
                        },
                    }
                )

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=False,
        )

        if streaming_callback:
            response_stream = self._model.create_chat_completion(
                messages=formatted_messages, tools=llamacpp_tools, **updated_generation_kwargs, stream=True
            )
            return self._handle_streaming_response(
                response_stream=response_stream,  # type: ignore[arg-type]
                streaming_callback=streaming_callback,
                component_info=ComponentInfo.from_component(self),
            )  # we know that response_stream is Iterator[CreateChatCompletionStreamResponse]
            # because create_chat_completion was called with stream=True, but mypy doesn't know that

        response = self._model.create_chat_completion(
            messages=formatted_messages, tools=llamacpp_tools, **updated_generation_kwargs
        )
        replies = []
        if not isinstance(response, dict):
            msg = f"Expected a dictionary response, got a different object: {response}"
            raise ValueError(msg)

        for choice in response["choices"]:
            chat_message = self._convert_chat_completion_choice_to_chat_message(choice, response)
            replies.append(chat_message)
        return {"replies": replies}

    @staticmethod
    def _handle_streaming_response(
        response_stream: Iterator[CreateChatCompletionStreamResponse],
        streaming_callback: SyncStreamingCallbackT,
        component_info: ComponentInfo,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Take streaming responses from llama.cpp, convert to Haystack StreamingChunk objects, stream them,
        and finally convert them to a ChatMessage.

        :param response_stream: The streaming responses from llama.cpp.
        :param streaming_callback: The callback function for streaming chunks.
        :param component_info: The component info.
        :returns: A dictionary with the replies.
        """
        streaming_chunks = []

        seen_tool_call_ids = set()  # Track tool call IDs we've seen

        for i, chunk in enumerate(response_stream):
            content = ""
            tool_calls = []
            mapped_finish_reason = None

            # Track new tool call IDs in this chunk.
            # Considering tool call ID is the only reliable way to recognize tool calls in llama.cpp streaming.
            # They are often spread across multiple chunks.
            new_tool_call_ids = set()

            if chunk.get("choices") and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                delta: Union[ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty, Dict] = (
                    choice.get("delta", {})
                )

                finish_reason = choice.get("finish_reason")
                mapped_finish_reason = FINISH_REASON_MAPPING.get(finish_reason or "")

                if content_raw := delta.get("content"):
                    content = str(content_raw)

                tool_calls_data = delta.get("tool_calls")
                if tool_calls_data is not None and isinstance(tool_calls_data, list):
                    for tool_call_chunk in tool_calls_data:
                        tool_call_id = tool_call_chunk.get("id")
                        is_new_tool_call = tool_call_id and tool_call_id not in seen_tool_call_ids

                        if is_new_tool_call:
                            new_tool_call_ids.add(tool_call_id)
                            seen_tool_call_ids.add(tool_call_id)

                        function_data = tool_call_chunk.get("function", {})

                        # Only include tool_name if this is a new tool call
                        tool_name = function_data.get("name", "") if is_new_tool_call else ""

                        tool_calls.append(
                            ToolCallDelta(
                                index=tool_call_chunk.get("index"),
                                id=tool_call_id,
                                tool_name=tool_name,
                                arguments=function_data.get("arguments"),
                            )
                        )

            # start is True if it's the first chunk or if we have new tool call IDs
            start = i == 0 or len(new_tool_call_ids) > 0

            streaming_chunk = StreamingChunk(
                content="" if tool_calls else content,  # prioritize tool calls over content when both are present
                tool_calls=tool_calls,
                component_info=component_info,
                index=i,
                start=start,
                finish_reason=mapped_finish_reason,
                meta={
                    "model": chunk["model"],
                    "received_at": datetime.fromtimestamp(chunk["created"], tz=timezone.utc).isoformat(),
                },  # llama.cpp does not provide usage metadata during streaming
            )

            streaming_chunks.append(streaming_chunk)

            # Stream the chunk
            try:
                streaming_callback(streaming_chunk)
            except Exception as e:
                logger.error(f"Error in streaming callback invocation: {e}")
                continue

        message = _convert_streaming_chunks_to_chat_message(streaming_chunks)
        return {"replies": [message]}

    @staticmethod
    def _convert_chat_completion_choice_to_chat_message(
        choice: ChatCompletionResponseChoice, response: CreateChatCompletionResponse
    ) -> ChatMessage:
        llamacpp_message = choice["message"]
        text_content = llamacpp_message.get("content", "") or None
        tool_calls = []

        if llamacpp_tool_calls := llamacpp_message.get("tool_calls", []):
            for llamacpp_tc in llamacpp_tool_calls:
                arguments_str = llamacpp_tc["function"]["arguments"]
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(
                        ToolCall(id=llamacpp_tc["id"], tool_name=llamacpp_tc["function"]["name"], arguments=arguments)
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "Llama.cpp returned a malformed JSON string for tool call arguments. This tool call "
                        "will be skipped. Tool call ID: {tc_id}, Tool name: {tc_name}, Arguments: {tc_args}",
                        tc_id=llamacpp_tc["id"],
                        tc_name=llamacpp_tc["function"]["name"],
                        tc_args=arguments_str,
                    )

        finish_reason = choice.get("finish_reason")

        meta = {
            "response_id": response["id"],
            "model": response["model"],
            "created": response["created"],
            "index": choice["index"],
            "finish_reason": FINISH_REASON_MAPPING.get(finish_reason or ""),
            "usage": response["usage"],
        }

        return ChatMessage.from_assistant(text=text_content, tool_calls=tool_calls, meta=meta)
