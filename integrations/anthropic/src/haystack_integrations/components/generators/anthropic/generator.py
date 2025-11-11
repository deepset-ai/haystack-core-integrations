from typing import Any, Callable, ClassVar, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

from anthropic import Anthropic, Stream
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
)

logger = logging.getLogger(__name__)


@component
class AnthropicGenerator:
    """
    Enables text generation using Anthropic large language models (LLMs). It supports the Claude family of models.

    Although Anthropic natively supports a much richer messaging API, we have intentionally simplified it in this
    component so that the main input/output interface is string-based.
    For more complete support, consider using the AnthropicChatGenerator.

    ```python
    from haystack_integrations.components.generators.anthropic import AnthropicGenerator

    client = AnthropicGenerator(model="claude-sonnet-4-20250514")
    response = client.run("What's Natural Language Processing? Be brief.")
    print(response)
    >>{'replies': ['Natural language processing (NLP) is a branch of artificial intelligence focused on enabling
    >>computers to understand, interpret, and manipulate human language. The goal of NLP is to read, decipher,
    >> understand, and make sense of the human languages in a manner that is valuable.'], 'meta': {'model':
    >> 'claude-2.1', 'index': 0, 'finish_reason': 'end_turn', 'usage': {'input_tokens': 18, 'output_tokens': 58}}}
    ```
    """

    # The parameters that can be passed to the Anthropic API https://docs.anthropic.com/claude/reference/messages_post
    ALLOWED_PARAMS: ClassVar[list[str]] = [
        "system",
        "max_tokens",
        "metadata",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
        "thinking",
    ]

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("ANTHROPIC_API_KEY"),  # noqa: B008
        model: str = "claude-sonnet-4-20250514",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize the AnthropicGenerator.

        :param api_key: The Anthropic API key.
        :param model: The name of the Anthropic model to use.
        :param streaming_callback: An optional callback function to handle streaming chunks.
        :param system_prompt: An optional system prompt to use for generation.
        :param generation_kwargs: Additional keyword arguments for generation.
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries

        self.include_thinking = self.generation_kwargs.pop("include_thinking", True)
        self.thinking_tag = self.generation_kwargs.pop("thinking_tag", "thinking")
        self.thinking_tag_start = f"<{self.thinking_tag}>" if self.thinking_tag else ""
        self.thinking_tag_end = f"</{self.thinking_tag}>\n\n" if self.thinking_tag else "\n\n"

        client_kwargs: dict[str, Any] = {"api_key": api_key.resolve_value()}
        # We do this since timeout=None is not the same as not setting it in Anthropic
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        # We do this since max_retries must be an int when passing to Anthropic
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = Anthropic(**client_kwargs)

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Get telemetry data for the component.

        :returns: A dictionary containing telemetry data.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            streaming_callback=callback_name,
            system_prompt=self.system_prompt,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnthropicGenerator":
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

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ) -> dict[str, Union[list[str], list[dict[str, Any]]]]:
        """
        Generate replies using the Anthropic API.

        :param prompt: The input prompt for generation.
        :param generation_kwargs: Additional keyword arguments for generation.
        :param streaming_callback: An optional callback function to handle streaming chunks.
        :returns: A dictionary containing:
         - `replies`: A list of generated replies.
         - `meta`: A list of metadata dictionaries for each reply.
        """
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        filtered_generation_kwargs = {k: v for k, v in generation_kwargs.items() if k in self.ALLOWED_PARAMS}
        disallowed_params = set(generation_kwargs) - set(self.ALLOWED_PARAMS)
        if disallowed_params:
            logger.warning(
                "Model parameters {disallowed_params} are not allowed and will be ignored. "
                "Allowed parameters are {allowed_params}.",
                disallowed_params=disallowed_params,
                allowed_params=self.ALLOWED_PARAMS,
            )

        streaming_callback = streaming_callback or self.streaming_callback
        stream = streaming_callback is not None
        response: Union[Message, Stream[MessageStreamEvent]] = self.client.messages.create(
            max_tokens=filtered_generation_kwargs.pop("max_tokens", 512),
            system=self.system_prompt if self.system_prompt else filtered_generation_kwargs.pop("system", ""),
            model=self.model,
            messages=[MessageParam(content=prompt, role="user")],
            stream=stream,
            **filtered_generation_kwargs,
        )

        completions: list[str] = []
        meta: dict[str, Any] = {}
        # if streaming is enabled, the response is a Stream[MessageStreamEvent]
        # some tracing libs (e.g. ddtrace) wrap the response breaking isinstance checks
        if stream:
            chunks: list[StreamingChunk] = []
            stream_event, delta, start_event, content_block_type = None, None, None, None
            for stream_event in response:
                if isinstance(stream_event, MessageStartEvent):
                    # capture start message to count input tokens
                    start_event = stream_event
                if isinstance(stream_event, ContentBlockDeltaEvent) and isinstance(stream_event.delta, TextDelta):
                    chunk_delta: StreamingChunk = StreamingChunk(content=stream_event.delta.text)
                    chunks.append(chunk_delta)
                    if streaming_callback:
                        streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
                if isinstance(stream_event, MessageDeltaEvent):
                    # capture stop reason and stop sequence
                    delta = stream_event
                if self.include_thinking:
                    if isinstance(stream_event, ContentBlockStartEvent):
                        content_block_type = stream_event.content_block.type
                        if content_block_type == "thinking":
                            start_tag_chunk = StreamingChunk(content=self.thinking_tag_start)
                            chunks.append(start_tag_chunk)
                            if streaming_callback:
                                streaming_callback(start_tag_chunk)
                    if isinstance(stream_event, ContentBlockStopEvent):
                        if content_block_type == "thinking":
                            end_tag_chunk = StreamingChunk(content=self.thinking_tag_end)
                            chunks.append(end_tag_chunk)
                            if streaming_callback:
                                streaming_callback(end_tag_chunk)
                    if isinstance(stream_event, ContentBlockDeltaEvent) and isinstance(
                        stream_event.delta, ThinkingDelta
                    ):
                        thinking_delta_chunk: StreamingChunk = StreamingChunk(content=stream_event.delta.thinking)
                        chunks.append(thinking_delta_chunk)
                        if streaming_callback:
                            streaming_callback(thinking_delta_chunk)

            completions = ["".join([chunk.content for chunk in chunks])]
            meta.update(
                {
                    "model": self.model,
                    "index": 0,
                    "finish_reason": delta.delta.stop_reason if delta else "end_turn",
                    "usage": {**dict(start_event.message.usage, **dict(delta.usage))} if delta and start_event else {},
                }
            )
        # if streaming is disabled, the response is an Anthropic Message
        elif isinstance(response, Message):
            completions = [
                content_block.text for content_block in response.content if isinstance(content_block, TextBlock)
            ]
            thinkings = [
                content_block.thinking for content_block in response.content if isinstance(content_block, ThinkingBlock)
            ]
            if self.include_thinking and len(thinkings) == len(completions):
                completions = [
                    f"{self.thinking_tag_start}{thinking}{self.thinking_tag_end}{completion}"
                    for thinking, completion in zip(thinkings, completions)
                ]

            meta.update(
                {
                    "model": response.model,
                    "index": 0,
                    "finish_reason": response.stop_reason,
                    "usage": dict(response.usage or {}),
                }
            )

        return {"replies": completions, "meta": [meta]}
