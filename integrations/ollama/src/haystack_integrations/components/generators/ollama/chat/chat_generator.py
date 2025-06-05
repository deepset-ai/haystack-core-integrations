from typing import Any, Callable, Dict, List, Literal, Optional, Union, DefaultDict
from collections import defaultdict

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


# --------------------------------------------------------------------------------------
# Helper conversions
# --------------------------------------------------------------------------------------
def _convert_chatmessage_to_ollama_format(message: ChatMessage) -> Dict[str, Any]:
    """
    Convert a ChatMessage to the format expected by the Ollama Chat API.
    """
    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results

    if not text_contents and not tool_calls and not tool_call_results:
        raise ValueError(
            "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
        )
    elif len(text_contents) + len(tool_call_results) > 1:
        raise ValueError("A `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`.")

    ollama_msg: Dict[str, Any] = {"role": message._role.value}

    if tool_call_results:
        # Ollama does not expose an error-field for failed tool invocations.
        ollama_msg["content"] = tool_call_results[0].result
        return ollama_msg

    if text_contents:
        ollama_msg["content"] = text_contents[0]
    if tool_calls:
        ollama_msg["tool_calls"] = [
            {"type": "function", "function": {"name": tc.tool_name, "arguments": tc.arguments}}
            for tc in tool_calls
        ]
    return ollama_msg


def _convert_ollama_meta_to_openai_format(input_response_dict: Dict) -> Dict:
    """
    Map Ollama metadata keys onto the OpenAI-compatible names Haystack expects.
    All unknown keys are preserved.
    """
    meta = {k: v for k, v in input_response_dict.items() if k != "message"}

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
    Convert **non-streaming** Ollama Chat API response to Haystack ChatMessage.
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


# --------------------------------------------------------------------------------------
# Main component
# --------------------------------------------------------------------------------------
@component
class OllamaChatGenerator:
    """
    Haystack generator for models served by **Ollama** (https://ollama.ai).

    * Fully supports streaming.
    * Correctly passes tool-calls to Haystack when `stream=True`.
    """

    # ------------------------------------------------------------------
    # Construction / (de)serialisation
    # ------------------------------------------------------------------
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
        :param model: Name of the model inside the running Ollama instance.
        :param url:   Base URL of that instance.
        :param generation_kwargs: Default inference-options (temperature, top_p …).
        :param timeout:  API timeout in seconds.
        :param keep_alive: Ollama “keep-alive” setting (seconds, duration string, -1 or '0').
        :param streaming_callback: Optional token callback when `stream=True`.
        :param tools: List[Tool] **or** Toolset; duplicate names are rejected.
        :param response_format: None / "json" / JSON Schema dict (Ollama ≥ 0.1.34).
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

    # Serialisation helpers
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            streaming_callback=serialize_callable(self.streaming_callback)
            if self.streaming_callback
            else None,
            tools=serialize_tools_or_toolset(self.tools),
            response_format=self.response_format,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaChatGenerator":
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        if callback_ser := data["init_parameters"].get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(callback_ser)
        return default_from_dict(cls, data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_chunk(chunk_response: Any) -> StreamingChunk:
        """
        Convert one Ollama stream-chunk to Haystack StreamingChunk.
        """
        d = chunk_response.model_dump()

        content = d["message"]["content"]
        meta = {k: v for k, v in d.items() if k != "message"}
        meta["role"] = d["message"]["role"]  # keep for debugging
        if tc := d["message"].get("tool_calls"):
            meta["tool_calls"] = tc

        return StreamingChunk(content, meta)

    def _handle_streaming_response(
        self,
        response_iter: Any,
        streaming_callback: Optional[Callable[[StreamingChunk], None]],
    ) -> Dict[str, List[ChatMessage]]:

        chunks: List[StreamingChunk] = []

        # For every tool-call id store EITHER
        #   * a concatenated str (when the model streams JSON snippets), OR
        #   * the most-recent dict (when it sends a full JSON object at once)
        tool_call_args_by_id: Dict[str, Union[str, dict]] = {}
        last_seen_calls: List[Dict[str, Any]] = []

        for raw_chunk in response_iter:
            chunk = self._build_chunk(raw_chunk)
            chunks.append(chunk)

            if streaming_callback is not None:
                streaming_callback(chunk)

            # ── collect tool-call deltas ──────────────────────────────
            for tc in chunk.meta.get("tool_calls", []):
                tc_id = tc.get("id") or tc["function"]["name"]
                arg_delta = tc["function"].get("arguments")

                if arg_delta is None:
                    continue

                prev = tool_call_args_by_id.get(tc_id)

                if isinstance(arg_delta, str):
                    if isinstance(prev, str):
                        tool_call_args_by_id[tc_id] = prev + arg_delta
                    else:  # prev is dict or None → start fresh with str
                        tool_call_args_by_id[tc_id] = arg_delta
                else:  # arg_delta is dict
                    tool_call_args_by_id[tc_id] = arg_delta  # overwrite; latest wins

                last_seen_calls.append(tc)

        # ── merge text ────────────────────────────────────────────────
        combined_text = "".join(c.content for c in chunks)

        # ── rebuild completed ToolCall objects ────────────────────────
        completed_tool_calls: List[ToolCall] = []
        if last_seen_calls:
            seen_ids = set()
            # walk backwards so earlier duplicates are skipped
            for tc in reversed(last_seen_calls):
                tc_id = tc.get("id") or tc["function"]["name"]
                if tc_id in seen_ids:
                    continue
                seen_ids.add(tc_id)

                completed_tool_calls.append(
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=tool_call_args_by_id.get(tc_id),
                    )
                )
            completed_tool_calls.reverse()  # restore original order

        reply = ChatMessage.from_assistant(
            text=combined_text,
            tool_calls=completed_tool_calls or None,
        )

        if chunks:
            reply._meta = _convert_ollama_meta_to_openai_format(chunks[-1].meta)

        return {"replies": [reply]}

    # ------------------------------------------------------------------
    # Main public entry point
    # ------------------------------------------------------------------
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
        Generate a reply given the full `messages` history.

        * If `streaming_callback` **or** the instance was created with one,
          streaming is enabled.
        * `tools` overrides the instance’s default list for this call only.
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        tools = tools or self.tools
        _check_duplicate_tool_names(tools)

        # Convert Toolset → list[Tool] for JSON serialisation
        if isinstance(tools, Toolset):
            tools = list(tools)
        ollama_tools = (
            [{"type": "function", "function": {**tool.tool_spec}} for tool in tools] if tools else None
        )

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
