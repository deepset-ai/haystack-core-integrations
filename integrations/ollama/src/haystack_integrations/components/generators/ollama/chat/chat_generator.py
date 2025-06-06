from typing import Any, Callable, Dict, List, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import (
    Tool,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.tools.toolset import Toolset
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from pydantic.json_schema import JsonSchemaValue

from ollama import ChatResponse, Client


def _convert_chatmessage_to_ollama_format(message: ChatMessage) -> Dict[str, Any]:
    """
    Convert a ChatMessage to the format expected by the Ollama Chat API.
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

    ollama_msg: Dict[str, Any] = {"role": message.role.value}

    if tool_call_results:
        # Ollama does not provide a way to communicate errors in tool invocations, so we ignore the error field
        ollama_msg["content"] = tool_call_results[0].result
        return ollama_msg

    if text_contents:
        ollama_msg["content"] = text_contents[0]
    if tool_calls:
        # Ollama does not support tool call id, so we ignore it
        ollama_msg["tool_calls"] = [
            {"type": "function", "function": {"name": tool_call.tool_name, "arguments": tool_call.arguments}}
            for tool_call in tool_calls
        ]
    return ollama_msg


def _convert_ollama_meta_to_openai_format(input_response_dict: Dict) -> Dict:
    """
    Map Ollama metadata keys onto the OpenAI-compatible names Haystack expects.
    All fields that are not part of the OpenAI metadata are left unchanged in the returned dict.

    Example Ollama metadata:
    {
        'model': 'phi4:14b-q4_K_M',
        'created_at': '2025-03-09T18:38:33.004185821Z',
        'done': True,
        'done_reason': 'stop',
        'total_duration': 86627206961,
        'load_duration': 23585622554,
        'prompt_eval_count': 26,
        'prompt_eval_duration': 3426000000,
        'eval_count': 298,
        'eval_duration': 4799921000
    }
    Example OpenAI metadata:
    {
        'model': 'phi4:14b-q4_K_M',
        'finish_reason': 'stop',
        'usage': {
            'completion_tokens': 298,
            'prompt_tokens': 26,
            'total_tokens': 324,
        }
        'completion_start_time': '2025-03-09T18:38:33.004185821Z',
        'done': True,
        'total_duration': 86627206961,
        'load_duration': 23585622554,
        'prompt_eval_duration': 3426000000,
        'eval_duration': 4799921000,
    }
    """
    meta = {key: value for key, value in input_response_dict.items() if key != "message"}

    if "done_reason" in meta:
        meta["finish_reason"] = meta.pop("done_reason")
    if "created_at" in meta:
        meta["completion_start_time"] = meta.pop("created_at")
    if "eval_count" in meta and "prompt_eval_count" in meta:
        eval_count = meta.pop("eval_count")
        prompt_eval_count = meta.pop("prompt_eval_count")
        meta["usage"] = {
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "total_tokens": eval_count + prompt_eval_count,
        }
    return meta


def _convert_ollama_response_to_chatmessage(ollama_response: "ChatResponse") -> ChatMessage:
    """
    Convert non-streaming Ollama Chat API response to Haystack ChatMessage with the assistant role.
    """
    response_dict = ollama_response.model_dump()
    ollama_message = response_dict["message"]

    text = ollama_message["content"]
    tool_calls: List[ToolCall] = []

    if ollama_tool_calls := ollama_message.get("tool_calls"):
        for ollama_tc in ollama_tool_calls:
            tool_calls.append(
                ToolCall(
                    tool_name=ollama_tc["function"]["name"],
                    arguments=ollama_tc["function"]["arguments"],
                )
            )

    chat_msg = ChatMessage.from_assistant(text=text, tool_calls=tool_calls or None)
    chat_msg._meta = _convert_ollama_meta_to_openai_format(response_dict)
    return chat_msg


@component
class OllamaChatGenerator:
    """
    Haystack generator for models served by Ollama (https://ollama.ai).

    * Fully supports streaming.
    * Correctly passes tool-calls to Haystack when `stream=True`.

    Usage example:
    ```python
    from haystack.components.generators.utils import print_streaming_chunk
    from haystack.components.agents import Agent
    from haystack_integrations.components.generators.ollama.chat import OllamaChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool

    def echo(query: str) -> str:
        print(f"Tool executed with QUERY: {query}")
        return query

    echo_tool = Tool(
        name="echo_tool",
        description="Echoes the query (demo tool).",
        function=echo,
        parameters={"query": {"type": "string", "description": "Search query"}},
    )
    agent = Agent(
        chat_generator=OllamaChatGenerator(model="mistral-small3.1:24b"),
        tools=[echo_tool],
        system_prompt=(
            "Use tool to print the query to test tools. Do not answer the question, just send the query to the tool"
        ),
        max_agent_steps=5,
        raise_on_tool_invocation_failure=True,
        streaming_callback=print_streaming_chunk,
    )
    messages = [ChatMessage.from_user("This is stream test of tool usage")]
    result = agent.run(messages=messages)
    for message in result["messages"]:
        print("\n======")
        if message.role == "system":
            continue
        elif message.role == "tool":
            print(f"{message.role}:")
            print(f"Tool Results: {[tool.result for tool in message.tool_call_results]}")
            print(f"Used Tools: {[tool.origin.tool_name for tool in message.tool_call_results]}\n")
        else:
            print(f"{message.role}: {message.text}")
            print(f"Used Tools: {[tool.tool_name for tool in message.tool_calls]}\n")
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
        tools: Optional[Union[List[Tool], Toolset]] = None,
        response_format: Optional[Union[None, Literal["json"], JsonSchemaValue]] = None,
    ):
        """
        :param model:
            The name of the model to use. The model must already be present (pulled) in the running Ollama instance.
        :param url:
            The base URL of the Ollama server (default "http://localhost:11434").
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param timeout:
            Socket timeout *in seconds* for HTTP calls to Ollama.
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
            A list of `haystack.tools.Tool` or a `haystack.tools.Toolset`. Duplicate tool names raise a `ValueError`.
            Not all models support tools. For a list of models compatible with tools, see the
            [models page](https://ollama.com/search?c=tools).
        :param response_format:
            The format for structured model outputs. The value can be:
            - None: No specific structure or format is applied to the response. The response is returned as-is.
            - "json": The response is formatted as a JSON object.
            - JSON Schema: The response is formatted as a JSON object
                that adheres to the specified JSON Schema. (needs Ollama ≥ 0.1.34)
        """
        _check_duplicate_tool_names(tools)

        self.model = model
        self.url = url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
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

        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            streaming_callback=callback_name,
            tools=serialize_tools_or_toolset(self.tools),
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
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        if callback_ser := data["init_parameters"].get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(callback_ser)
        return default_from_dict(cls, data)

    @staticmethod
    def _build_chunk(chunk_response: Any) -> StreamingChunk:
        """
        Convert one Ollama stream-chunk to Haystack StreamingChunk.
        """
        chunk_response_dict = chunk_response.model_dump()

        content = chunk_response_dict["message"]["content"]
        meta = {key: value for key, value in chunk_response_dict.items() if key != "message"}
        meta["role"] = chunk_response_dict["message"]["role"]
        if tool_calls := chunk_response_dict["message"].get("tool_calls"):
            meta["tool_calls"] = tool_calls

        return StreamingChunk(content, meta)

    def _handle_streaming_response(
        self,
        response_iter: Any,
        callback: Optional[Callable[[StreamingChunk], None]],
    ) -> Dict[str, List[ChatMessage]]:
        """
        Merge an Ollama streaming response into a single ChatMessage, preserving
        tool calls.  Works even when arguments arrive piecemeal as str fragments
        or as full JSON dicts.
        """

        chunks: List[StreamingChunk] = []

        # Accumulators
        arg_by_id: Dict[str, Union[str, dict]] = {}
        name_by_id: Dict[str, str] = {}
        id_order: List[str] = []

        # Stream
        for raw in response_iter:
            chunk = self._build_chunk(raw)
            chunks.append(chunk)

            if callback:
                callback(chunk)

            for tool_call in chunk.meta.get("tool_calls", []):
                # the Ollama server doesn't guarantee an id field in every tool_calls entry.
                # OpenAI-compatible endpoint (/v1/chat/completions) - recent releases do add an auto-generated id
                # when the model produces multiple tool calls, so that clients can map results back.
                # Native Ollama endpoint (/api/chat) and older builds
                # - the JSON often contains only function.name + arguments;
                # many users have reported that id is missing even with several calls,
                # making client-side resolution harder:
                # https://github.com/ollama/ollama/issues/6708
                # https://github.com/ollama/ollama/issues/7510
                # - If id is provided → we can distinguish multiple calls to the same tool.
                # - If id is missing → fallback to function.name works only when there's one call.
                # - That's why the deduplication logic is cautious and assumes one logical
                #   call per name when id is absent.
                tool_call_id = tool_call.get("id") or tool_call["function"]["name"]
                args = tool_call["function"].get("arguments")

                # Remember first-seen order and tool name
                if tool_call_id not in id_order:
                    id_order.append(tool_call_id)
                    name_by_id[tool_call_id] = tool_call["function"]["name"]

                # Update the argument accumulator for this tool_call_id.
                #
                # • Ollama may stream the same `arguments` field in *two* different forms:
                #   1) as one or more **str fragments**  -- characters of a JSON string, delivered chunk-by-chunk;
                #   2) as a complete **dict**            -- fully-parsed JSON in a single chunk.
                #
                # • A dict always represents the *final* state (it is already parsed JSON),
                #   so it should **overwrite** anything collected before.
                #
                # • If we are still receiving str fragments *and* we have not yet seen a
                #   dict, we concatenate them in arrival order.
                #
                # • If a dict has already been stored (`prev` is dict) and another string
                #   fragment arrives, we ignore it by skipping the concat - this prevents
                #   `TypeError: can only concatenate str (not "dict") to str` and keeps the
                #   fully-formed JSON intact.
                if isinstance(args, dict):
                    # Dict beats anything seen so far (final, authoritative version).
                    arg_by_id[tool_call_id] = args
                elif isinstance(args, str):
                    # Append only when we are still in "string mode".
                    if not isinstance(arg_by_id.get(tool_call_id), dict):
                        prev = arg_by_id.get(tool_call_id, "")
                        arg_by_id[tool_call_id] = f"{prev}{args}"

        # Compose final reply
        text = "".join(c.content for c in chunks)

        tool_calls = [
            ToolCall(tool_name=name_by_id[tool_call_id], arguments=arg_by_id.get(tool_call_id))
            for tool_call_id in id_order
        ]

        reply = ChatMessage.from_assistant(
            text=text,
            tool_calls=tool_calls or None,
            meta=_convert_ollama_meta_to_openai_format(chunks[-1].meta) if chunks else None,
        )
        return {"replies": [reply]}

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        *,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Runs an Ollama Model on a given chat history.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Per-call overrides for Ollama inference options.
            These are merged on top of the instance-level `generation_kwargs`.
            Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance. If set, it will override the `tools` parameter set
            during component initialization.
        :param streaming_callback:
            A callable to receive `StreamingChunk` objects as they
            arrive.  Supplying a callback (here or in the constructor) switches
            the component into streaming mode.
        :returns: A dictionary with the following keys:
            - `replies`: A list of ChatMessages containing the model's response
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        tools = tools or self.tools
        _check_duplicate_tool_names(tools)

        # Convert Toolset → list[Tool] for JSON serialization
        if isinstance(tools, Toolset):
            tools = list(tools)
        ollama_tools = [{"type": "function", "function": {**tool.tool_spec}} for tool in tools] if tools else None

        callback = streaming_callback or self.streaming_callback
        is_stream = callback is not None

        ollama_messages = [_convert_chatmessage_to_ollama_format(m) for m in messages]

        response = self._client.chat(
            model=self.model,
            messages=ollama_messages,
            tools=ollama_tools,
            stream=is_stream,
            keep_alive=self.keep_alive,
            options=generation_kwargs,
            format=self.response_format,
        )

        if is_stream:
            return self._handle_streaming_response(response, callback)

        # non-stream path
        return {"replies": [_convert_ollama_response_to_chatmessage(response)]}
