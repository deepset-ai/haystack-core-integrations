import json
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.lazy_imports import LazyImport
from haystack.tools import (
    Tool,
    Toolset,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from cohere import ChatResponse

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    import cohere

logger = logging.getLogger(__name__)


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
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def _format_message(message: ChatMessage) -> Dict[str, Any]:
    """
    Formats a Haystack ChatMessage into Cohere's chat format.

    The function handles message components including:
    - Text content
    - Tool calls
    - Tool call results

    :param message: Haystack ChatMessage to format.
    :return: Dictionary formatted according to Cohere's chat specification.
    """
    if not message.texts and not message.tool_calls and not message.tool_call_results:
        msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
        raise ValueError(msg)

    cohere_msg: Dict[str, Any] = {"role": message.role.value}

    # Format the message based on its content type
    if message.tool_call_results:
        result = message.tool_call_results[0]  # We expect one result at a time
        if result.origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Cohere."
            raise ValueError(msg)
        cohere_msg.update(
            {
                "role": "tool",
                "tool_call_id": result.origin.id,
                "content": [{"type": "document", "document": {"data": json.dumps({"result": result.result})}}],
            }
        )
    elif message.tool_calls:
        tool_calls = []
        for tool_call in message.tool_calls:
            if tool_call.id is None:
                msg = "`ToolCall` must have a non-null `id` attribute to be used with Cohere."
                raise ValueError(msg)
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.tool_name, "arguments": json.dumps(tool_call.arguments)},
                }
            )
        cohere_msg.update(
            {
                "tool_calls": tool_calls,
                "tool_plan": message.text if message.text else "",
            }
        )
    else:
        cohere_msg["content"] = (
            [{"type": "text", "text": message.texts[0]}] if message.texts and message.texts[0] else []
        )
        if not cohere_msg["content"]:
            msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
            raise ValueError(msg)

    return cohere_msg


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
        # Convert Cohere tool calls to Haystack ToolCall objects
        tool_calls = [
            ToolCall(id=tc.id, tool_name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in chat_response.message.tool_calls
        ]
        # Create message with tool plan as text and tool calls in the format Haystack expects
        tool_plan = chat_response.message.tool_plan or ""
        message = ChatMessage.from_assistant(text=tool_plan, tool_calls=tool_calls)
    elif chat_response.message.content:
        message = ChatMessage.from_assistant(chat_response.message.content[0].text)
    else:
        # Handle the case where neither tool_calls nor content exists
        logger.warning(f"Received empty response from Cohere API: {chat_response.message}")
        message = ChatMessage.from_assistant("")

    # In V2, token usage is part of the response object, not the message
    message.meta.update(
        {
            "model": model,
            "usage": {
                "prompt_tokens": (chat_response.usage.billed_units.input_tokens),
                "completion_tokens": (chat_response.usage.billed_units.output_tokens),
            },
            "index": 0,
            "finish_reason": chat_response.finish_reason,
            "citations": chat_response.message.citations,
        }
    )
    return message


def _parse_streaming_response(
    response: Generator, model: str, streaming_callback: Callable[[StreamingChunk], None]
) -> ChatMessage:
    """
    Parses Cohere's streaming chat response into a Haystack ChatMessage.

    Processes streaming chunks and aggregates them into a complete response,
    including:
    - Text content
    - Tool plan
    - Tool calls and their arguments
    - Usage statistics
    - Finish reason

    :param response: Streaming response from Cohere's chat API.
    :param model: The name of the model that generated the response.
    :param streaming_callback: Callback function for streaming chunks.
    :return: A Haystack ChatMessage containing the formatted response.
    """
    response_text = ""
    tool_plan = ""
    tool_calls = []
    current_tool_call = None
    current_tool_arguments = ""
    captured_meta = {}

    for chunk in response:
        if chunk and chunk.type == "content-delta":
            stream_chunk = StreamingChunk(content=chunk.delta.message.content.text)
            streaming_callback(stream_chunk)
            response_text += chunk.delta.message.content.text
        elif chunk and chunk.type == "tool-plan-delta":
            tool_plan += chunk.delta.message.tool_plan
            stream_chunk = StreamingChunk(content=chunk.delta.message.tool_plan)
            streaming_callback(stream_chunk)
        elif chunk and chunk.type == "tool-call-start":
            tool_call = chunk.delta.message.tool_calls
            current_tool_call = ToolCall(id=tool_call.id, tool_name=tool_call.function.name, arguments="")
        elif chunk and chunk.type == "tool-call-delta":
            current_tool_arguments += chunk.delta.message.tool_calls.function.arguments
        elif chunk and chunk.type == "tool-call-end":
            if current_tool_call:
                current_tool_call.arguments = json.loads(current_tool_arguments)
                tool_calls.append(current_tool_call)
                current_tool_call = None
                current_tool_arguments = ""
        elif chunk and chunk.type == "message-end":
            captured_meta.update(
                {
                    "model": model,
                    "index": 0,
                    "finish_reason": chunk.delta.finish_reason,
                    "usage": {
                        "prompt_tokens": chunk.delta.usage.billed_units.input_tokens,
                        "completion_tokens": chunk.delta.usage.billed_units.output_tokens,
                    },
                }
            )

    # Create the appropriate ChatMessage based on what we received
    if tool_calls:
        chat_message = ChatMessage.from_assistant(text=tool_plan, tool_calls=tool_calls)
    else:
        chat_message = ChatMessage.from_assistant(text=response_text)

    # Add metadata
    chat_message.meta.update(captured_meta)

    return chat_message


@component
class CohereChatGenerator:
    """
    Completes chats using Cohere's models using Cohere cohere.ClientV2 `chat` endpoint.

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
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        **kwargs,
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
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should have a unique name.

        """
        cohere_import.check()
        _check_duplicate_tool_names(list(tools or []))  # handles Toolset as well

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
        self.client = cohere.ClientV2(
            api_key=self.api_key.resolve_value(), base_url=self.api_base_url, client_name="haystack"
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
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Invoke the chat endpoint based on the provided messages and generation parameters.

        :param messages: list of `ChatMessage` instances representing the input messages.
        :param generation_kwargs: additional keyword arguments for chat generation. These parameters will
            potentially override the parameters passed in the __init__ method.
            For more details on the parameters supported by the Cohere API, refer to the
            Cohere [documentation](https://docs.cohere.com/reference/chat).
        :param tools: A list of tools or a Toolset for which the model can prepare calls. If set, it will override
            the `tools` parameter set during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: a list of `ChatMessage` instances representing the generated responses.
        """
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # Handle tools
        tools = tools or self.tools
        if isinstance(tools, Toolset):
            tools = list(tools)
        if tools:
            _check_duplicate_tool_names(tools)
            generation_kwargs["tools"] = [_format_tool(tool) for tool in tools]

        formatted_messages = [_format_message(message) for message in messages]

        if self.streaming_callback:
            response = self.client.chat_stream(
                model=self.model,
                messages=formatted_messages,
                **generation_kwargs,
            )
            chat_message = _parse_streaming_response(response, self.model, self.streaming_callback)
        else:
            response = self.client.chat(
                model=self.model,
                messages=formatted_messages,
                **generation_kwargs,
            )
            chat_message = _parse_response(response, self.model)

        return {"replies": [chat_message]}
