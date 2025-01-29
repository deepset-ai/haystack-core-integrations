from typing import Any, Callable, Dict, List, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool, _check_duplicate_tool_names, deserialize_tools_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from pydantic.json_schema import JsonSchemaValue

from ollama import ChatResponse, Client


def _convert_chatmessage_to_ollama_format(message: ChatMessage) -> Dict[str, Any]:
    """
    Convert a ChatMessage to the format expected by Ollama Chat API.
    """
    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results

    if not text_contents and not tool_calls and not tool_call_results:
        msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
        raise ValueError(msg)
    elif len(text_contents) + len(tool_call_results) > 1:
        msg = "A `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`."
        raise ValueError(msg)

    ollama_msg: Dict[str, Any] = {"role": message._role.value}

    if tool_call_results:
        # Ollama does not provide a way to communicate errors in tool invocations, so we ignore the error field
        ollama_msg["content"] = tool_call_results[0].result
        return ollama_msg

    if text_contents:
        ollama_msg["content"] = text_contents[0]
    if tool_calls:
        # Ollama does not support tool call id, so we ignore it
        ollama_msg["tool_calls"] = [
            {"type": "function", "function": {"name": tc.tool_name, "arguments": tc.arguments}} for tc in tool_calls
        ]
    return ollama_msg


def _convert_ollama_response_to_chatmessage(ollama_response: "ChatResponse") -> ChatMessage:
    """
    Converts the non-streaming response from the Ollama API to a ChatMessage with assistant role.
    """
    response_dict = ollama_response.model_dump()

    ollama_message = response_dict["message"]

    text = ollama_message["content"]

    tool_calls = []
    if ollama_tool_calls := ollama_message.get("tool_calls"):
        for ollama_tc in ollama_tool_calls:
            tool_calls.append(
                ToolCall(tool_name=ollama_tc["function"]["name"], arguments=ollama_tc["function"]["arguments"])
            )

    message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls)

    message.meta.update({key: value for key, value in response_dict.items() if key != "message"})
    return message


@component
class OllamaChatGenerator:
    """
    Supports models running on Ollama, such as llama2 and mixtral.  Find the full list of supported models
    [here](https://ollama.ai/library).

    Usage example:
    ```python
    from haystack_integrations.components.generators.ollama import OllamaChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = OllamaChatGenerator(model="zephyr",
                                url = "http://localhost:11434",
                                generation_kwargs={
                                "num_predict": 100,
                                "temperature": 0.9,
                                })

    messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
    ChatMessage.from_user("What's Natural Language Processing?")]

    print(generator.run(messages=messages))
    ```
    """

    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        keep_alive: Optional[Union[float, str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Union[None, Literal["json"], JsonSchemaValue]] = None,
    ):
        """
        :param model:
            The name of the model to use. The model should be available in the running Ollama instance.
        :param url:
            The URL of a running Ollama instance.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param keep_alive:
            The option that controls how long the model will stay loaded into memory following the request.
            If not set, it will use the default value from the Ollama (5 minutes).
            The value can be set to:
            - a duration string (such as "10m" or "24h")
            - a number in seconds (such as 3600)
            - any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            - '0' which will unload the model immediately after generating a response.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param tools:
            A list of tools for which the model can prepare calls.
            Not all models support tools. For a list of models compatible with tools, see the
            [models page](https://ollama.com/search?c=tools).
        :param response_format:
            The format for structured model outputs. The value can be:
            - None: No specific structure or format is applied to the response. The response is returned as-is.
            - "json": The response is formatted as a JSON object.
            - JSON Schema: The response is formatted as a JSON object that adheres to the specified JSON Schema.
        """

        _check_duplicate_tool_names(tools)

        self.timeout = timeout
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model
        self.keep_alive = keep_alive
        self.streaming_callback = streaming_callback
        self.tools = tools
        self.response_format = response_format
        self._client = Client(host=self.url, timeout=self.timeout)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
              Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        serialized_tools = [tool.to_dict() for tool in self.tools] if self.tools else None
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            keep_alive=self.keep_alive,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            streaming_callback=callback_name,
            tools=serialized_tools,
            response_format=self.response_format,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_tools_inplace(data["init_parameters"], key="tools")

        init_params = data.get("init_parameters", {})

        if serialized_callback_handler := init_params.get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _build_chunk(self, chunk_response: Any) -> StreamingChunk:
        """
        Converts the response from the Ollama API to a StreamingChunk.
        """
        chunk_response_dict = chunk_response.model_dump()

        content = chunk_response_dict["message"]["content"]
        meta = {key: value for key, value in chunk_response_dict.items() if key != "message"}
        meta["role"] = chunk_response_dict["message"]["role"]

        chunk_message = StreamingChunk(content, meta)
        return chunk_message

    def _handle_streaming_response(self, response) -> Dict[str, List[Any]]:
        """
        Handles streaming response and converts it to Haystack format
        """
        chunks: List[StreamingChunk] = []
        for chunk in response:
            chunk_delta = self._build_chunk(chunk)
            chunks.append(chunk_delta)
            if self.streaming_callback is not None:
                self.streaming_callback(chunk_delta)

        replies = [ChatMessage.from_assistant("".join([c.content for c in chunks]))]
        meta = {key: value for key, value in chunks[0].meta.items() if key != "message"}

        return {"replies": replies, "meta": [meta]}

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Runs an Ollama Model on a given chat history.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        stream = self.streaming_callback is not None
        tools = tools or self.tools
        _check_duplicate_tool_names(tools)

        if stream and tools:
            msg = "Ollama does not support tools and streaming at the same time. Please choose one."
            raise ValueError(msg)

        if self.response_format and tools:
            msg = "Ollama does not support tools and response_format at the same time. Please choose one."
            raise ValueError(msg)

        if self.response_format and stream:
            msg = "Ollama does not support streaming and response_format at the same time. Please choose one."
            raise ValueError(msg)

        ollama_tools = [{"type": "function", "function": {**t.tool_spec}} for t in tools] if tools else None

        ollama_messages = [_convert_chatmessage_to_ollama_format(msg) for msg in messages]
        response = self._client.chat(
            model=self.model,
            messages=ollama_messages,
            tools=ollama_tools,
            stream=stream,
            keep_alive=self.keep_alive,
            options=generation_kwargs,
            format=self.response_format,
        )

        if stream:
            return self._handle_streaming_response(response)

        return {"replies": [_convert_ollama_response_to_chatmessage(response)]}
