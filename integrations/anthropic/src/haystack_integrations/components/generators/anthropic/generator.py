from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

from anthropic import Anthropic, Stream
from anthropic.types import (
    ContentBlockDeltaEvent,
    Message,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStreamEvent,
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

    client = AnthropicGenerator(model="claude-3-sonnet-20240229")
    response = client.run("What's Natural Language Processing? Be brief.")
    print(response)
    >>{'replies': ['Natural language processing (NLP) is a branch of artificial intelligence focused on enabling
    >>computers to understand, interpret, and manipulate human language. The goal of NLP is to read, decipher,
    >> understand, and make sense of the human languages in a manner that is valuable.'], 'meta': {'model':
    >> 'claude-2.1', 'index': 0, 'finish_reason': 'end_turn', 'usage': {'input_tokens': 18, 'output_tokens': 58}}}
    ```
    """

    # The parameters that can be passed to the Anthropic API https://docs.anthropic.com/claude/reference/messages_post
    ALLOWED_PARAMS: ClassVar[List[str]] = [
        "system",
        "max_tokens",
        "metadata",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
    ]

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("ANTHROPIC_API_KEY"),  # noqa: B008
        model: str = "claude-3-sonnet-20240229",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
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
        self.client = Anthropic(api_key=self.api_key.resolve_value())

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Get telemetry data for the component.

        :returns: A dictionary containing telemetry data.
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
            system_prompt=self.system_prompt,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnthropicGenerator":
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

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Generate replies using the Anthropic API.

        :param prompt: The input prompt for generation.
        :param generation_kwargs: Additional keyword arguments for generation.
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
                f"Model parameters {disallowed_params} are not allowed and will be ignored. "
                f"Allowed parameters are {self.ALLOWED_PARAMS}."
            )

        response: Union[Message, Stream[MessageStreamEvent]] = self.client.messages.create(
            max_tokens=filtered_generation_kwargs.pop("max_tokens", 512),
            system=self.system_prompt if self.system_prompt else filtered_generation_kwargs.pop("system", ""),
            model=self.model,
            messages=[MessageParam(content=prompt, role="user")],
            stream=self.streaming_callback is not None,
            **filtered_generation_kwargs,
        )

        completions: List[str] = []
        meta: Dict[str, Any] = {}
        # if streaming is enabled, the response is a Stream[MessageStreamEvent]
        if isinstance(response, Stream):
            chunks: List[StreamingChunk] = []
            stream_event, delta, start_event = None, None, None
            for stream_event in response:
                if isinstance(stream_event, MessageStartEvent):
                    # capture start message to count input tokens
                    start_event = stream_event
                if isinstance(stream_event, ContentBlockDeltaEvent):
                    chunk_delta: StreamingChunk = StreamingChunk(content=stream_event.delta.text)
                    chunks.append(chunk_delta)
                    if self.streaming_callback:
                        self.streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
                if isinstance(stream_event, MessageDeltaEvent):
                    # capture stop reason and stop sequence
                    delta = stream_event
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
            completions = [content_block.text for content_block in response.content]
            meta.update(
                {
                    "model": response.model,
                    "index": 0,
                    "finish_reason": response.stop_reason,
                    "usage": dict(response.usage or {}),
                }
            )

        return {"replies": completions, "meta": [meta]}
