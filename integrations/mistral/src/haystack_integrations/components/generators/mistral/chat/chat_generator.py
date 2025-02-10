# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Callable, Dict, List, Optional

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)


@component
class MistralChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Mistral AI generative models.
    For supported models, see [Mistral AI docs](https://docs.mistral.ai/platform/endpoints/#operation/listModels).

    Users can pass any text generation parameters valid for the Mistral Chat Completion API
    directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the Mistral API Chat Completion endpoint.
    - **Streaming Support**: Supports streaming responses from the Mistral API Chat Completion endpoint.
    - **Customizability**: Supports all parameters supported by the Mistral API Chat Completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage)

    For more details on the parameters supported by the Mistral API, refer to the
    [Mistral API Docs](https://docs.mistral.ai/api/).

    Usage example:
    ```python
    from haystack_integrations.components.generators.mistral import MistralChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = MistralChatGenerator()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
    >>meta={'model': 'mistral-small-latest', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-small-latest",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = "https://api.mistral.ai/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Creates an instance of MistralChatGenerator. Unless specified otherwise in the `model`, this is for Mistral's
        `mistral-small-latest` model.

        :param api_key:
            The Mistral API key.
        :param model:
            The name of the Mistral chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Mistral API Base url.
            For more details, see Mistral [docs](https://docs.mistral.ai/api/).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Mistral endpoint. See [Mistral API docs](https://docs.mistral.ai/api/) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `safe_prompt`: Whether to inject a safety prompt before all conversations.
            - `random_seed`: The seed to use for random sampling.
        :param tools:
            A list of tools for which the model can prepare calls.
        """
        super(MistralChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
            tools=tools,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        # if we didn't implement the to_dict method here then the to_dict method of the superclass would be used
        # which would serialiaze some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=[tool.to_dict() for tool in self.tools] if self.tools else None,
        )

    def _convert_streaming_chunks_to_chat_message(self, chunk: Any, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.

        :param chunk: The last chunk returned by the OpenAI API.
        :param chunks: The list of all `StreamingChunk` objects.
        """

        # to have streaming support and tool calls we need to do some extra work here because the superclass
        # looks for tool calls in the first chunk only, while Mistral does not return tool calls in the first chunk
        # so we need to find the chunk that has tool calls and use it to create the ChatMessage
        # after we implement https://github.com/deepset-ai/haystack/pull/8829 we'll be able to remove this code
        # and use the superclass implementation
        text = "".join([chunk.content for chunk in chunks])
        tool_calls = []

        # are there any tool calls in the chunks?
        if any(chunk.meta.get("tool_calls") for chunk in chunks):
            payloads = {}  # Use a dict to track tool calls by ID
            for chunk_payload in chunks:
                deltas = chunk_payload.meta.get("tool_calls") or []

                # deltas is a list of ChoiceDeltaToolCall
                for delta in deltas:
                    if delta.id not in payloads:
                        payloads[delta.id] = {"id": delta.id, "arguments": "", "name": "", "type": None}
                    # ChoiceDeltaToolCall has a 'function' field of type ChoiceDeltaToolCallFunction
                    if delta.function:
                        # For tool calls with the same ID, use the latest values
                        if delta.function.name is not None:
                            payloads[delta.id]["name"] = delta.function.name
                        if delta.function.arguments is not None:
                            # Use the latest arguments value
                            payloads[delta.id]["arguments"] = delta.function.arguments
                    if delta.type is not None:
                        payloads[delta.id]["type"] = delta.type

            for payload in payloads.values():
                arguments_str = payload["arguments"]
                try:
                    # Try to parse the concatenated arguments string as JSON
                    arguments = json.loads(arguments_str)
                    tool_calls.append(ToolCall(id=payload["id"], tool_name=payload["name"], arguments=arguments))
                except json.JSONDecodeError:
                    logger.warning(
                        "Mistral returned a malformed JSON string for tool call arguments. This tool call "
                        "will be skipped. Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                        _id=payload["id"],
                        _name=payload["name"],
                        _arguments=arguments_str,
                    )

        meta = {
            "model": chunk.model,
            "index": 0,
            "finish_reason": chunk.choices[0].finish_reason,
            "completion_start_time": chunks[0].meta.get("received_at"),  # first chunk received
            "usage": {},  # we don't have usage data for streaming responses
        }

        return ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta)
