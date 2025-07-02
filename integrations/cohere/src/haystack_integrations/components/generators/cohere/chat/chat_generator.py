import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    ToolCall,
)
from haystack.dataclasses.streaming_chunk import (
    AsyncStreamingCallbackT,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    select_streaming_callback,
)
from haystack.tools import (
    Tool,
    Toolset,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import (
    deserialize_callable,
    serialize_callable,
)

from cohere import (
    AssistantChatMessageV2,
    AsyncClientV2,
    ChatResponse,
    ClientV2,
    StreamedChatResponseV2,
    SystemChatMessageV2,
    TextAssistantMessageContentItem,
    TextContent,
    TextSystemMessageContentItem,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    UserChatMessageV2,
)

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
    - Tool calls
    - Tool call results

    :param message: Haystack ChatMessage to format.
    :return: A Cohere message object.
    """
    if not message.texts and not message.tool_calls and not message.tool_call_results:
        msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
        raise ValueError(msg)

    # Format the message based on its content type
    if message.tool_call_results:
        result = message.tool_call_results[0]  # We expect one result at a time
        if result.origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Cohere."
            raise ValueError(msg)
        return ToolChatMessageV2(tool_call_id=result.origin.id, content=json.dumps({"result": result.result}))
    elif message.tool_calls:
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
    else:
        if not message.texts or not message.texts[0]:
            msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
            raise ValueError(msg)

        if message.role.value == "user":
            return UserChatMessageV2(content=[TextContent(text=message.texts[0])])
        elif message.role.value == "assistant":
            return AssistantChatMessageV2(content=[TextAssistantMessageContentItem(text=message.texts[0])])
        elif message.role.value == "system":
            return SystemChatMessageV2(content=[TextSystemMessageContentItem(text=message.texts[0])])
        else:
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
    elif chat_response.message.content:
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


def _initialize_streaming_state():
    """Initialize the state variables for streaming response parsing."""
    return {
        "response_text": "",
        "tool_plan": "",
        "tool_calls": [],
        "current_tool_call": None,
        "current_tool_arguments": "",
        "captured_meta": {},
    }


def _process_cohere_chunk(cohere_chunk: StreamedChatResponseV2, state: Dict[str, Any], model: str) -> Optional[str]:
    """
    Process a single streamed chat response and update the parsing state.

    :param cohere_chunk: The streamed chat response from Cohere's API
    :param state: Dictionary containing the current parsing state
    :param model: Model name for metadata
    :return: Content to stream (if any), None otherwise
    """
    if not cohere_chunk:
        return None

    if not hasattr(cohere_chunk, "delta") or cohere_chunk.delta is None:
        return None

    if cohere_chunk.type == "content-delta":
        if (
            cohere_chunk.delta.message
            and cohere_chunk.delta.message.content
            and cohere_chunk.delta.message.content.text is not None
        ):
            content = cohere_chunk.delta.message.content.text
            state["response_text"] += content
            return content
    elif cohere_chunk.type == "tool-plan-delta":
        if cohere_chunk.delta.message and cohere_chunk.delta.message.tool_plan is not None:
            content = cohere_chunk.delta.message.tool_plan
            state["tool_plan"] += content
            return content
    elif cohere_chunk.type == "tool-call-start":
        if cohere_chunk.delta.message and cohere_chunk.delta.message.tool_calls:
            tool_call = cohere_chunk.delta.message.tool_calls
            function = tool_call.function
            if function is not None and function.name is not None and isinstance(function.name, str):
                state["current_tool_call"] = ToolCall(
                    id=tool_call.id,
                    tool_name=function.name,
                    arguments={},
                )
                # Reset arguments for new tool call
                state["current_tool_arguments"] = ""
    elif cohere_chunk.type == "tool-call-delta":
        if (
            cohere_chunk.delta.message
            and cohere_chunk.delta.message.tool_calls
            and cohere_chunk.delta.message.tool_calls.function
            and cohere_chunk.delta.message.tool_calls.function.arguments is not None
        ):
            state["current_tool_arguments"] += cohere_chunk.delta.message.tool_calls.function.arguments
    elif cohere_chunk.type == "tool-call-end":
        if state["current_tool_call"]:
            try:
                if state["current_tool_arguments"].strip():
                    state["current_tool_call"].arguments = json.loads(state["current_tool_arguments"])
                state["tool_calls"].append(state["current_tool_call"])
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse tool call arguments: {e}")
            finally:
                state["current_tool_call"] = None
                state["current_tool_arguments"] = ""
    elif cohere_chunk.type == "message-end":
        # Handle any remaining tool call that wasn't properly ended
        if state["current_tool_call"]:
            try:
                if state["current_tool_arguments"].strip():
                    state["current_tool_call"].arguments = json.loads(state["current_tool_arguments"])
                state["tool_calls"].append(state["current_tool_call"])
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse tool call arguments at message end: {e}")
            finally:
                state["current_tool_call"] = None
                state["current_tool_arguments"] = ""

        if (
            cohere_chunk.delta.finish_reason is not None
            and cohere_chunk.delta.usage
            and cohere_chunk.delta.usage.billed_units
            and cohere_chunk.delta.usage.billed_units.input_tokens is not None
            and cohere_chunk.delta.usage.billed_units.output_tokens is not None
        ):
            state["captured_meta"].update(
                {
                    "model": model,
                    "index": 0,
                    "finish_reason": cohere_chunk.delta.finish_reason,
                    "usage": {
                        "prompt_tokens": cohere_chunk.delta.usage.billed_units.input_tokens,
                        "completion_tokens": cohere_chunk.delta.usage.billed_units.output_tokens,
                    },
                }
            )

    return None


def _finalize_streaming_message(state):
    """
    Create a ChatMessage from the final parsing state.

    :param state: Dictionary containing the parsed state
    :return: ChatMessage with metadata
    """
    # Create the appropriate ChatMessage based on what we received
    if state["tool_calls"]:
        chat_message = ChatMessage.from_assistant(text=state["tool_plan"], tool_calls=state["tool_calls"])
    else:
        chat_message = ChatMessage.from_assistant(text=state["response_text"])

    # Add metadata
    chat_message._meta.update(state["captured_meta"])
    return chat_message


def _parse_streaming_response(
    response: Iterator[StreamedChatResponseV2],
    model: str,
    streaming_callback: SyncStreamingCallbackT,
    component_info: ComponentInfo,
) -> ChatMessage:
    """
    Parses Cohere's streaming chat response into a Haystack ChatMessage.
    """
    state = _initialize_streaming_state()

    for chunk in response:
        stream_content = _process_cohere_chunk(cohere_chunk=chunk, state=state, model=model)
        if stream_content:
            stream_chunk = StreamingChunk(content=stream_content, component_info=component_info)
            streaming_callback(stream_chunk)

    return _finalize_streaming_message(state)


async def _parse_async_streaming_response(
    response: AsyncIterator[StreamedChatResponseV2],
    model: str,
    streaming_callback: AsyncStreamingCallbackT,
    component_info: ComponentInfo,
) -> ChatMessage:
    """
    Parses Cohere's async streaming chat response into a Haystack ChatMessage.
    """
    state = _initialize_streaming_state()

    async for chunk in response:
        stream_content = _process_cohere_chunk(cohere_chunk=chunk, state=state, model=model)
        if stream_content:
            stream_chunk = StreamingChunk(content=stream_content, component_info=component_info)
            await streaming_callback(stream_chunk)

    return _finalize_streaming_message(state)


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
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
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
        :param tools: A list of Tool objects or a Toolset that the model can use. Each tool should have a unique name.

        """
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
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ) -> Dict[str, List[ChatMessage]]:
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
        generation_kwargs = {
            **self.generation_kwargs,
            **(generation_kwargs or {}),
        }

        # Handle tools
        tools = tools or self.tools
        if isinstance(tools, Toolset):
            tools = list(tools)
        if tools:
            _check_duplicate_tool_names(tools)
            generation_kwargs["tools"] = [_format_tool(tool) for tool in tools]

        formatted_messages = [_format_message(message) for message in messages]

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=None, requires_async=False
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
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Asynchronously invoke the chat endpoint based on the provided messages and generation parameters.

        :param messages: list of `ChatMessage` instances representing the input messages.
        :param generation_kwargs: additional keyword arguments for chat generation. These parameters will
            potentially override the parameters passed in the __init__ method.
            For more details on the parameters supported by the Cohere API, refer to the
            Cohere [documentation](https://docs.cohere.com/reference/chat).
        :param tools: A list of tools for which the model can prepare calls. If set, it will override
            the `tools` parameter set during component initialization.
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
        if isinstance(tools, Toolset):
            tools = list(tools)
        if tools:
            _check_duplicate_tool_names(tools)
            generation_kwargs["tools"] = [_format_tool(tool) for tool in tools]

        formatted_messages = [_format_message(message) for message in messages]

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=None, requires_async=True
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
