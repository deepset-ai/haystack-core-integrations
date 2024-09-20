import dataclasses
import json
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

from anthropic import Anthropic, Stream
from anthropic.types import (
    ContentBlockDeltaEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


@component
class AnthropicChatGenerator:
    """
    Enables text generation using Anthropic state-of-the-art Claude 3 family of large language models (LLMs) through
    the Anthropic messaging API.

    It supports models like `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-sonnet`, and `claude-3-haiku`,
    accessed through the [`/v1/messages`](https://docs.anthropic.com/en/api/messages) API endpoint.

    Users can pass any text generation parameters valid for the Anthropic messaging API directly to this component
    via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs` parameter in the `run` method.

    For more details on the parameters supported by the Anthropic API, refer to the
    Anthropic Message API [documentation](https://docs.anthropic.com/en/api/messages).

    ```python
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]
    client = AnthropicChatGenerator(model="claude-3-5-sonnet-20240620")
    response = client.run(messages)
    print(response)

    >> {'replies': [ChatMessage(content='Natural Language Processing (NLP) is a field of artificial intelligence that
    >> focuses on enabling computers to understand, interpret, and generate human language. It involves developing
    >> techniques and algorithms to analyze and process text or speech data, allowing machines to comprehend and
    >> communicate in natural languages like English, Spanish, or Chinese.', role=<ChatRole.ASSISTANT: 'assistant'>,
    >> name=None, meta={'model': 'claude-3-5-sonnet-20240620', 'index': 0, 'finish_reason': 'end_turn',
    >> 'usage': {'input_tokens': 15, 'output_tokens': 64}})]}
    ```

    For more details on supported models and their capabilities, refer to the Anthropic
    [documentation](https://docs.anthropic.com/claude/docs/intro-to-claude).

    Note: We only support text input/output modalities, and
    image [modality](https://docs.anthropic.com/en/docs/build-with-claude/vision) is not supported in
    this version of AnthropicChatGenerator.
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
    ]

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("ANTHROPIC_API_KEY"),  # noqa: B008
        model: str = "claude-3-5-sonnet-20240620",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ignore_tools_thinking_messages: bool = True,
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
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.client = Anthropic(api_key=self.api_key.resolve_value())
        self.ignore_tools_thinking_messages = ignore_tools_thinking_messages

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
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
                                  potentially override the parameters passed in the `__init__` method.
                                  For more details on the parameters supported by the Anthropic API, refer to the
                                  Anthropic [documentation](https://www.anthropic.com/python-library).

        :returns:
            - `replies`: A list of ChatMessage instances representing the generated responses.
        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        filtered_generation_kwargs = {k: v for k, v in generation_kwargs.items() if k in self.ALLOWED_PARAMS}
        disallowed_params = set(generation_kwargs) - set(self.ALLOWED_PARAMS)
        if disallowed_params:
            logger.warning(
                f"Model parameters {disallowed_params} are not allowed and will be ignored. "
                f"Allowed parameters are {self.ALLOWED_PARAMS}."
            )
        system_messages: List[ChatMessage] = [msg for msg in messages if msg.is_from(ChatRole.SYSTEM)]
        non_system_messages: List[ChatMessage] = [msg for msg in messages if not msg.is_from(ChatRole.SYSTEM)]
        system_messages_formatted: List[Dict[str, Any]] = (
            self._convert_to_anthropic_format(system_messages) if system_messages else []
        )
        messages_formatted: List[Dict[str, Any]] = (
            self._convert_to_anthropic_format(non_system_messages) if non_system_messages else []
        )

        extra_headers = filtered_generation_kwargs.get("extra_headers", {})
        prompt_caching_on = "anthropic-beta" in extra_headers and "prompt-caching" in extra_headers["anthropic-beta"]
        has_cached_messages = any("cache_control" in m for m in system_messages_formatted) or any(
            "cache_control" in m for m in messages_formatted
        )
        if has_cached_messages and not prompt_caching_on:
            # this avoids Anthropic errors when prompt caching is not enabled
            # but user requested individual messages to be cached
            logger.warn(
                "Prompt caching is not enabled but you requested individual messages to be cached. "
                "Messages will be sent to the API without prompt caching."
            )
            system_messages_formatted = list(map(self._remove_cache_control, system_messages_formatted))
            messages_formatted = list(map(self._remove_cache_control, messages_formatted))

        response: Union[Message, Stream[MessageStreamEvent]] = self.client.messages.create(
            max_tokens=filtered_generation_kwargs.pop("max_tokens", 512),
            system=system_messages_formatted or filtered_generation_kwargs.pop("system", ""),
            model=self.model,
            messages=messages_formatted,
            stream=self.streaming_callback is not None,
            **filtered_generation_kwargs,
        )

        completions: List[ChatMessage] = []
        # if streaming is enabled, the response is a Stream[MessageStreamEvent]
        if isinstance(response, Stream):
            chunks: List[StreamingChunk] = []
            stream_event, delta, start_event = None, None, None
            for stream_event in response:
                if isinstance(stream_event, MessageStartEvent):
                    # capture start message to count input tokens
                    start_event = stream_event
                if isinstance(stream_event, ContentBlockDeltaEvent):
                    chunk_delta: StreamingChunk = self._build_chunk(stream_event.delta)
                    chunks.append(chunk_delta)
                    if self.streaming_callback:
                        self.streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
                if isinstance(stream_event, MessageDeltaEvent):
                    # capture stop reason and stop sequence
                    delta = stream_event
            completions = [self._connect_chunks(chunks, start_event, delta)]

        # if streaming is disabled, the response is an Anthropic Message
        elif isinstance(response, Message):
            has_tools_msgs = any(isinstance(content_block, ToolUseBlock) for content_block in response.content)
            if has_tools_msgs and self.ignore_tools_thinking_messages:
                response.content = [block for block in response.content if isinstance(block, ToolUseBlock)]
            completions = [self._build_message(content_block, response) for content_block in response.content]

        # rename the meta key to be inline with OpenAI meta output keys
        for response in completions:
            if response.meta is not None and "usage" in response.meta:
                response.meta["usage"]["prompt_tokens"] = response.meta["usage"].pop("input_tokens")
                response.meta["usage"]["completion_tokens"] = response.meta["usage"].pop("output_tokens")

        return {"replies": completions}

    def _build_message(self, content_block: Union[TextBlock, ToolUseBlock], message: Message) -> ChatMessage:
        """
        Converts the non-streaming Anthropic Message to a ChatMessage.
        :param content_block: The content block of the message.
        :param message: The non-streaming Anthropic Message.
        :returns: The ChatMessage.
        """
        if isinstance(content_block, TextBlock):
            chat_message = ChatMessage.from_assistant(content_block.text)
        else:
            chat_message = ChatMessage.from_assistant(json.dumps(content_block.model_dump(mode="json")))
        chat_message.meta.update(
            {
                "model": message.model,
                "index": 0,
                "finish_reason": message.stop_reason,
                "usage": dict(message.usage or {}),
            }
        )
        return chat_message

    def _convert_to_anthropic_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Converts the list of ChatMessage to the list of messages in the format expected by the Anthropic API.
        :param messages: The list of ChatMessage.
        :returns: The list of messages in the format expected by the Anthropic API.
        """
        anthropic_formatted_messages = []
        for m in messages:
            message_dict = dataclasses.asdict(m)
            formatted_message = {k: v for k, v in message_dict.items() if k in {"role", "content"} and v}
            if m.is_from(ChatRole.SYSTEM):
                # system messages are treated differently and MUST be in the format expected by the Anthropic API
                # remove role and content from the message dict, add type and text
                formatted_message.pop("role")
                formatted_message["type"] = "text"
                formatted_message["text"] = formatted_message.pop("content")
            formatted_message.update(m.meta or {})
            anthropic_formatted_messages.append(formatted_message)
        return anthropic_formatted_messages

    def _connect_chunks(
        self, chunks: List[StreamingChunk], message_start: MessageStartEvent, delta: MessageDeltaEvent
    ) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.
        :param chunks: The list of all chunks returned by the Anthropic API.
        :param message_start: The MessageStartEvent.
        :param delta: The MessageDeltaEvent.
        :returns: The complete ChatMessage.
        """
        complete_response = ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))
        complete_response.meta.update(
            {
                "model": self.model,
                "index": 0,
                "finish_reason": delta.delta.stop_reason if delta else "end_turn",
                "usage": {**dict(message_start.message.usage, **dict(delta.usage))} if delta and message_start else {},
            }
        )
        return complete_response

    def _build_chunk(self, delta: TextDelta) -> StreamingChunk:
        """
        Converts the ContentBlockDeltaEvent to a StreamingChunk.
        :param delta: The ContentBlockDeltaEvent.
        :returns: The StreamingChunk.
        """
        return StreamingChunk(content=delta.text)

    def _remove_cache_control(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Removes the cache_control key from the message.
        :param message: The message to remove the cache_control key from.
        :returns: The message with the cache_control key removed.
        """
        return {k: v for k, v in message.items() if k != "cache_control"}
