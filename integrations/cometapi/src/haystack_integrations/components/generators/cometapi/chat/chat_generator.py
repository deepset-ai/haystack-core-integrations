from datetime import datetime
from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat import openai as openai_module
from haystack.components.generators.utils import _serialize_object
from haystack.dataclasses import (
    ComponentInfo,
    FinishReason,
    StreamingCallbackT,
    StreamingChunk,
    ToolCallDelta,
)
from haystack.tools import (
    Tool,
    Toolset,
)
from haystack.utils import Secret
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice


# TODO: remove the following function and the monkey patch after haystack-ai==2.23.0, that ships this fix in the base
# class
def _convert_chat_completion_chunk_to_streaming_chunk(
    chunk: ChatCompletionChunk, previous_chunks: list[StreamingChunk], component_info: ComponentInfo | None = None
) -> StreamingChunk:
    """
    Converts the streaming response chunk from the OpenAI API to a StreamingChunk.

    :param chunk: The chunk returned by the OpenAI API.
    :param previous_chunks: A list of previously received StreamingChunks.
    :param component_info: An optional `ComponentInfo` object containing information about the component that
        generated the chunk, such as the component name and type.

    :returns:
        A StreamingChunk object representing the content of the chunk from the OpenAI API.
    """
    finish_reason_mapping: dict[str, FinishReason] = {
        "stop": "stop",
        "length": "length",
        "content_filter": "content_filter",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
    }
    # On very first chunk so len(previous_chunks) == 0, the Choices field only provides role info (e.g. "assistant")
    # Choices is empty if include_usage is set to True where the usage information is returned.
    if len(chunk.choices) == 0:
        return StreamingChunk(
            content="",
            component_info=component_info,
            # Index is None since it's only set to an int when a content block is present
            index=None,
            finish_reason=None,
            meta={
                "model": chunk.model,
                "received_at": datetime.now().isoformat(),  # noqa: DTZ005
                "usage": _serialize_object(chunk.usage),
            },
        )

    choice: ChunkChoice = chunk.choices[0]

    # create a list of ToolCallDelta objects from the tool calls
    if choice.delta and choice.delta.tool_calls:
        tool_calls_deltas = []
        for tool_call in choice.delta.tool_calls:
            function = tool_call.function
            tool_calls_deltas.append(
                ToolCallDelta(
                    index=tool_call.index,
                    id=tool_call.id,
                    tool_name=function.name if function else None,
                    arguments=function.arguments if function and function.arguments else None,
                )
            )
        chunk_message = StreamingChunk(
            content=choice.delta.content or "",
            component_info=component_info,
            # We adopt the first tool_calls_deltas.index as the overall index of the chunk.
            index=tool_calls_deltas[0].index,
            tool_calls=tool_calls_deltas,
            start=tool_calls_deltas[0].tool_name is not None,
            finish_reason=finish_reason_mapping.get(choice.finish_reason) if choice.finish_reason else None,
            meta={
                "model": chunk.model,
                "index": choice.index,
                "tool_calls": choice.delta.tool_calls,
                "finish_reason": choice.finish_reason,
                "received_at": datetime.now().isoformat(),  # noqa: DTZ005
                "usage": _serialize_object(chunk.usage),
            },
        )
        return chunk_message

    # On very first chunk the choice field only provides role info (e.g. "assistant") so we set index to None
    # We set all chunks missing the content field to index of None. E.g. can happen if chunk only contains finish
    # reason.
    if choice.delta and (choice.delta.content is None or choice.delta.role is not None):
        resolved_index = None
    else:
        # We set the index to be 0 since if text content is being streamed then no tool calls are being streamed
        # NOTE: We may need to revisit this if OpenAI allows planning/thinking content before tool calls like
        #       Anthropic Claude
        resolved_index = 0

    # Initialize meta dictionary
    meta = {
        "model": chunk.model,
        "index": choice.index,
        "tool_calls": choice.delta.tool_calls if choice.delta and choice.delta.tool_calls else None,
        "finish_reason": choice.finish_reason,
        "received_at": datetime.now().isoformat(),  # noqa: DTZ005
        "usage": _serialize_object(chunk.usage),
    }

    # check if logprobs are present
    # logprobs are returned only for text content
    logprobs = _serialize_object(choice.logprobs) if choice.logprobs else None
    if logprobs:
        meta["logprobs"] = logprobs

    content = ""
    if choice.delta and choice.delta.content:
        content = choice.delta.content

    chunk_message = StreamingChunk(
        content=content,
        component_info=component_info,
        index=resolved_index,
        # The first chunk is always a start message chunk that only contains role information, so if we reach here
        # and previous_chunks is length 1 then this is the start of text content.
        start=len(previous_chunks) == 1,
        finish_reason=finish_reason_mapping.get(choice.finish_reason) if choice.finish_reason else None,
        meta=meta,
    )
    return chunk_message


# monkey patch the OpenAIChatGenerator to use our own _convert_chat_completion_chunk_to_streaming_chunk
openai_module._convert_chat_completion_chunk_to_streaming_chunk = _convert_chat_completion_chunk_to_streaming_chunk


class CometAPIChatGenerator(OpenAIChatGenerator):
    """
    A chat generator that uses the CometAPI for generating chat responses.

    This class extends Haystack's OpenAIChatGenerator to specifically interact with the CometAPI.
    It sets the `api_base_url` to the CometAPI endpoint and allows for all the
    standard configurations available in the OpenAIChatGenerator.

    :param api_key: The API key for authenticating with the CometAPI. Defaults to
                    loading from the "COMET_API_KEY" environment variable.
    :param model: The name of the model to use for chat generation (e.g., "gpt-5-mini", "grok-3-mini").
                  Defaults to "gpt-5-mini".
    :param streaming_callback: An optional callable that will be called with each chunk of
                                a streaming response.
    :param generation_kwargs: Optional keyword arguments to pass to the underlying generation
                              API call.
    :param timeout: The maximum time in seconds to wait for a response from the API.
    :param max_retries: The maximum number of times to retry a failed API request.
    :param tools: An optional list of tool definitions that the model can use.
    :param tools_strict: If True, the model is forced to use one of the provided tools if a tool call is made.
    :param http_client_kwargs: Optional keyword arguments to pass to the HTTP client.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("COMET_API_KEY"),
        model: str = "gpt-5-mini",
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        tools: list[Tool | Toolset] | Toolset | None = None,
        tools_strict: bool = False,
        http_client_kwargs: dict[str, Any] | None = None,
    ):
        api_base_url = "https://api.cometapi.com/v1"

        super().__init__(
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            tools=tools,
            tools_strict=tools_strict,
            http_client_kwargs=http_client_kwargs,
        )
