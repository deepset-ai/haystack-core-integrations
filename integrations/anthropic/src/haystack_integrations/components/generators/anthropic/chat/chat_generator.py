import json
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    StreamingCallbackT,
    StreamingChunk,
    ToolCall,
    ToolCallResult,
    select_streaming_callback,
)
from haystack.tools import (
    Tool,
    Toolset,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

from anthropic import Anthropic, AsyncAnthropic

logger = logging.getLogger(__name__)


def _update_anthropic_message_with_tool_call_results(
    tool_call_results: List[ToolCallResult], anthropic_msg: Dict[str, Any]
) -> None:
    """
    Update an Anthropic message with tool call results.

    :param tool_call_results: The list of ToolCallResults to update the message with.
    :param anthropic_msg: The Anthropic message to update.
    """
    if "content" not in anthropic_msg:
        anthropic_msg["content"] = []

    for tool_call_result in tool_call_results:
        if tool_call_result.origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Anthropic."
            raise ValueError(msg)
        anthropic_msg["content"].append(
            {
                "type": "tool_result",
                "tool_use_id": tool_call_result.origin.id,
                "content": [{"type": "text", "text": tool_call_result.result}],
                "is_error": tool_call_result.error,
            }
        )


def _convert_tool_calls_to_anthropic_format(tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
    """
    Convert a list of tool calls to the format expected by Anthropic Chat API.

    :param tool_calls: The list of ToolCalls to convert.
    :return: A list of dictionaries in the format expected by Anthropic API.
    """
    anthropic_tool_calls = []
    for tc in tool_calls:
        if tc.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Anthropic."
            raise ValueError(msg)
        anthropic_tool_calls.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.tool_name,
                "input": tc.arguments,
            }
        )
    return anthropic_tool_calls


def _convert_messages_to_anthropic_format(
    messages: List[ChatMessage],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert a list of messages to the format expected by Anthropic Chat API.

    :param messages: The list of ChatMessages to convert.
    :return: A tuple of two lists:
        - A list of system message dictionaries in the format expected by Anthropic API.
        - A list of non-system message dictionaries in the format expected by Anthropic API.
    """

    anthropic_system_messages = []
    anthropic_non_system_messages = []

    i = 0
    while i < len(messages):
        message = messages[i]

        # allow passing cache_control
        cache_control = {"cache_control": message.meta.get("cache_control")} if "cache_control" in message.meta else {}

        # system messages have special format requirements for Anthropic API
        # they can have only type and text fields, and they need to be passed separately
        # to the Anthropic API endpoint
        if message.is_from(ChatRole.SYSTEM):
            anthropic_system_messages.append({"type": "text", "text": message.text, **cache_control})
            i += 1
            continue

        anthropic_msg: Dict[str, Any] = {"role": message._role.value, "content": [], **cache_control}

        if message.texts and message.texts[0]:
            anthropic_msg["content"].append({"type": "text", "text": message.texts[0]})
        if message.tool_calls:
            anthropic_msg["content"] += _convert_tool_calls_to_anthropic_format(message.tool_calls)

        if message.tool_call_results:
            results = message.tool_call_results.copy()
            # Handle consecutive tool call results
            while (i + 1) < len(messages) and messages[i + 1].tool_call_results:
                i += 1
                results.extend(messages[i].tool_call_results)

            _update_anthropic_message_with_tool_call_results(results, anthropic_msg)
            anthropic_msg["role"] = "user"

        if not anthropic_msg["content"]:
            msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
            raise ValueError(msg)

        anthropic_non_system_messages.append(anthropic_msg)
        i += 1

    return anthropic_system_messages, anthropic_non_system_messages


@component
class AnthropicChatGenerator:
    """
    Completes chats using Anthropic's large language models (LLMs).

    It uses [ChatMessage](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
    format in input and output.

    You can customize how the text is generated by passing parameters to the
    Anthropic API. Use the `**generation_kwargs` argument when you initialize
    the component or when you run it. Any parameter that works with
    `anthropic.Message.create` will work here too.

    For details on Anthropic API parameters, see
    [Anthropic documentation](https://docs.anthropic.com/en/api/messages).

    Usage example:
    ```python
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = AnthropicChatGenerator(model="claude-3-5-sonnet-20240620",
                                       generation_kwargs={
                                           "max_tokens": 1000,
                                           "temperature": 0.7,
                                       })

    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]
    print(generator.run(messages=messages))
    """

    # The parameters that can be passed to the Anthropic API https://docs.anthropic.com/claude/reference/messages_post
    ALLOWED_PARAMS: ClassVar[List[str]] = [
        "system",
        "tools",
        "tool_choice",
        "max_tokens",
        "metadata",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
        "extra_headers",
        "thinking",
    ]

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("ANTHROPIC_API_KEY"),  # noqa: B008
        model: str = "claude-3-5-sonnet-20240620",
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ignore_tools_thinking_messages: bool = True,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Creates an instance of AnthropicChatGenerator.

        :param api_key: The Anthropic API key
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the Anthropic endpoint. See Anthropic [documentation](https://docs.anthropic.com/claude/reference/messages_post)
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
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should have a unique name.

        """
        _check_duplicate_tool_names(list(tools or []))  # handles Toolset as well

        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.client = Anthropic(api_key=self.api_key.resolve_value())
        self.async_client = AsyncAnthropic(api_key=self.api_key.resolve_value())
        self.ignore_tools_thinking_messages = ignore_tools_thinking_messages
        self.tools = tools

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            ignore_tools_thinking_messages=self.ignore_tools_thinking_messages,
            tools=serialize_tools_or_toolset(self.tools),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnthropicChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)

    @staticmethod
    def _get_openai_compatible_usage(response_dict: dict) -> dict:
        """
        Converts Anthropic usage metadata to OpenAI compatible format.
        """
        usage = response_dict.get("usage", {})
        if usage:
            if "input_tokens" in usage:
                usage["prompt_tokens"] = usage.pop("input_tokens")
            if "output_tokens" in usage:
                usage["completion_tokens"] = usage.pop("output_tokens")

        return usage

    def _convert_chat_completion_to_chat_message(
        self, anthropic_response: Any, ignore_tools_thinking_messages: bool
    ) -> ChatMessage:
        """
        Converts the response from the Anthropic API to a ChatMessage.
        """
        tool_calls = [
            ToolCall(tool_name=block.name, arguments=block.input, id=block.id)
            for block in anthropic_response.content
            if block.type == "tool_use"
        ]

        # Extract and join text blocks, respecting ignore_tools_thinking_messages
        text = ""
        if not (ignore_tools_thinking_messages and tool_calls):
            text = " ".join(block.text for block in anthropic_response.content if block.type == "text")

        message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls)

        # Dump the chat completion to a dict
        response_dict = anthropic_response.model_dump()
        usage = self._get_openai_compatible_usage(response_dict)
        message._meta.update(
            {
                "model": response_dict.get("model", None),
                "index": 0,
                "finish_reason": response_dict.get("stop_reason", None),
                "usage": usage,
            }
        )
        return message

    @staticmethod
    def _convert_anthropic_chunk_to_streaming_chunk(chunk: Any) -> StreamingChunk:
        """
        Converts an Anthropic StreamEvent to a StreamingChunk.
        """
        content = ""
        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
            content = chunk.delta.text

        return StreamingChunk(content=content, meta=chunk.model_dump())

    def _convert_streaming_chunks_to_chat_message(
        self, chunks: List[StreamingChunk], model: Optional[str] = None
    ) -> ChatMessage:
        """
        Converts a list of StreamingChunks to a ChatMessage.
        """
        full_content = ""
        tool_calls = []
        current_tool_call: Optional[Dict[str, Any]] = {}

        # loop through chunks and call the appropriate handler
        for chunk in chunks:
            chunk_type = chunk.meta.get("type")
            if chunk_type == "content_block_start":
                if chunk.meta.get("content_block", {}).get("type") == "tool_use":
                    delta_block = chunk.meta.get("content_block")
                    current_tool_call = {
                        "id": delta_block.get("id"),
                        "name": delta_block.get("name"),
                        "arguments": "",
                    }
            elif chunk_type == "content_block_delta":
                delta = chunk.meta.get("delta", {})
                if delta.get("type") == "text_delta":
                    full_content += delta.get("text", "")
                elif delta.get("type") == "input_json_delta" and current_tool_call:
                    current_tool_call["arguments"] += delta.get("partial_json", "")
            elif chunk_type == "message_delta":
                if chunk.meta.get("delta", {}).get("stop_reason") == "tool_use" and current_tool_call:
                    try:
                        # arguments is a string, convert to json
                        tool_calls.append(
                            ToolCall(
                                id=current_tool_call.get("id"),
                                tool_name=str(current_tool_call.get("name")),
                                arguments=json.loads(current_tool_call.get("arguments", {})),
                            )
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            "Anthropic returned a malformed JSON string for tool call arguments. "
                            "This tool call will be skipped. Arguments: {current_arguments}",
                            current_arguments=current_tool_call.get("arguments", ""),
                        )
                    current_tool_call = None

        message = ChatMessage.from_assistant(full_content, tool_calls=tool_calls)

        # Update meta information
        last_chunk_meta = chunks[-1].meta
        usage = self._get_openai_compatible_usage(last_chunk_meta)
        message._meta.update(
            {
                "model": model,
                "index": 0,
                "finish_reason": last_chunk_meta.get("delta", {}).get("stop_reason", None),
                "usage": usage,
            }
        )

        return message

    @staticmethod
    def _remove_cache_control(message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Removes the cache_control key from the message.
        :param message: The message to remove the cache_control key from.
        :returns: The message with the cache_control key removed.
        """
        return {k: v for k, v in message.items() if k != "cache_control"}

    def _prepare_request_params(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Prepare the parameters for the Anthropic API request.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Optional arguments to pass to the Anthropic generation endpoint.
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should
        have a unique name.
        :returns: A tuple containing:
            - system_messages: List of system messages in Anthropic format
            - non_system_messages: List of non-system messages in Anthropic format
            - generation_kwargs: Processed generation kwargs
            - anthropic_tools: List of tools in Anthropic format
        """
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        disallowed_params = set(generation_kwargs) - set(self.ALLOWED_PARAMS)
        if disallowed_params:
            logger.warning(
                "Model parameters {disallowed_params} are not allowed and will be ignored. "
                "Allowed parameters are {allowed_params}.",
                disallowed_params=disallowed_params,
                allowed_params=self.ALLOWED_PARAMS,
            )
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if k in self.ALLOWED_PARAMS}

        system_messages, non_system_messages = _convert_messages_to_anthropic_format(messages)

        # prompt caching
        extra_headers = generation_kwargs.get("extra_headers", {})
        prompt_caching_on = "anthropic-beta" in extra_headers and "prompt-caching" in extra_headers["anthropic-beta"]
        has_cached_messages = any("cache_control" in m for m in system_messages) or any(
            "cache_control" in m for m in non_system_messages
        )
        if has_cached_messages and not prompt_caching_on:
            # this avoids Anthropic errors when prompt caching is not enabled
            # but user requested individual messages to be cached
            logger.warn(
                "Prompt caching is not enabled but you requested individual messages to be cached. "
                "Messages will be sent to the API without prompt caching."
            )
            system_messages = list(map(self._remove_cache_control, system_messages))
            non_system_messages = list(map(self._remove_cache_control, non_system_messages))

        # tools management
        tools = tools or self.tools
        tools = list(tools) if isinstance(tools, Toolset) else tools
        _check_duplicate_tool_names(tools)  # handles Toolset as well
        anthropic_tools = (
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]
            if tools
            else []
        )

        return system_messages, non_system_messages, generation_kwargs, anthropic_tools

    def _process_response(
        self,
        response: Any,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Process the response from the Anthropic API.

        :param response: The response from the Anthropic API.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :returns: A dictionary containing the processed response as a list of ChatMessage objects.
        """
        # workaround for https://github.com/DataDog/dd-trace-py/issues/12562
        stream = streaming_callback is not None
        if stream:
            chunks: List[StreamingChunk] = []
            model: Optional[str] = None
            for chunk in response:
                if chunk.type == "message_start":
                    model = chunk.message.model
                elif chunk.type in [
                    "content_block_start",
                    "content_block_delta",
                    "message_delta",
                ]:
                    streaming_chunk = self._convert_anthropic_chunk_to_streaming_chunk(chunk)
                    chunks.append(streaming_chunk)
                    if streaming_callback:
                        streaming_callback(streaming_chunk)

            completion = self._convert_streaming_chunks_to_chat_message(chunks, model)
            return {"replies": [completion]}
        else:
            return {
                "replies": [
                    self._convert_chat_completion_to_chat_message(response, self.ignore_tools_thinking_messages)
                ]
            }

    async def _process_response_async(
        self,
        response: Any,
        streaming_callback: Optional[AsyncStreamingCallbackT] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Process the response from the Anthropic API asynchronously.

        :param response: The response from the Anthropic API.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.

        :returns:
            A dictionary containing the processed response as a list of ChatMessage objects.
        """
        # workaround for https://github.com/DataDog/dd-trace-py/issues/12562
        stream = streaming_callback is not None
        if stream:
            chunks: List[StreamingChunk] = []
            model: Optional[str] = None
            async for chunk in response:
                if chunk.type == "message_start":
                    model = chunk.message.model
                elif chunk.type in [
                    "content_block_start",
                    "content_block_delta",
                    "message_delta",
                ]:
                    streaming_chunk = self._convert_anthropic_chunk_to_streaming_chunk(chunk)
                    chunks.append(streaming_chunk)
                    if streaming_callback:
                        await streaming_callback(streaming_chunk)

            completion = self._convert_streaming_chunks_to_chat_message(chunks, model)
            return {"replies": [completion]}
        else:
            return {
                "replies": [
                    self._convert_chat_completion_to_chat_message(response, self.ignore_tools_thinking_messages)
                ]
            }

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Invokes the Anthropic API with the given messages and generation kwargs.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Optional arguments to pass to the Anthropic generation endpoint.
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should
        have a unique name. If set, it will override the `tools` parameter set during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        system_messages, non_system_messages, generation_kwargs, anthropic_tools = self._prepare_request_params(
            messages, generation_kwargs, tools
        )

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=False,
        )

        response = self.client.messages.create(
            model=self.model,
            messages=non_system_messages,
            system=system_messages,
            tools=anthropic_tools,
            stream=streaming_callback is not None,
            max_tokens=generation_kwargs.pop("max_tokens", 1024),
            **generation_kwargs,
        )

        return self._process_response(response, streaming_callback)

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Async version of the run method. Invokes the Anthropic API with the given messages and generation kwargs.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Optional arguments to pass to the Anthropic generation endpoint.
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should
        have a unique name. If set, it will override the `tools` parameter set during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        system_messages, non_system_messages, generation_kwargs, anthropic_tools = self._prepare_request_params(
            messages, generation_kwargs, tools
        )

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=True,
        )

        response = await self.async_client.messages.create(
            model=self.model,
            messages=non_system_messages,
            system=system_messages,
            tools=anthropic_tools,
            stream=streaming_callback is not None,
            max_tokens=generation_kwargs.pop("max_tokens", 1024),
            **generation_kwargs,
        )

        return await self._process_response_async(response, streaming_callback)
