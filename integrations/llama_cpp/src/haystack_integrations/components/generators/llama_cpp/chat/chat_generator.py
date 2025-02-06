import json
import logging
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, _check_duplicate_tool_names, deserialize_tools_inplace
from llama_cpp import ChatCompletionResponseChoice, CreateChatCompletionResponse, Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

logger = logging.getLogger(__name__)


def _convert_message_to_llamacpp_format(message: ChatMessage) -> Dict[str, Any]:
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

    role = message._role.value
    if role == "tool":
        role = "function"

    llamacpp_msg: Dict[str, Any] = {"role": role}

    if tool_call_results:
        if tool_call_results[0].origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with llama.cpp."
            raise ValueError(msg)
        llamacpp_msg["content"] = tool_call_results[0].result
        llamacpp_msg["tool_call_id"] = tool_call_results[0].origin.id
        # Llama.cpp does not provide a way to communicate errors in tool invocations, so we ignore the error field
        return llamacpp_msg

    if text_contents:
        llamacpp_msg["content"] = text_contents[0]
    if tool_calls:
        llamacpp_tool_calls = []
        for tc in tool_calls:
            if tc.id is None:
                msg = "`ToolCall` must have a non-null `id` attribute to be used with llama.cpp."
                raise ValueError(msg)
            llamacpp_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    # We disable ensure_ascii so special chars like emojis are not converted
                    "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
            )
        llamacpp_msg["tool_calls"] = llamacpp_tool_calls
    return llamacpp_msg


@component
class LlamaCppChatGenerator:
    """
    Provides an interface to generate text using LLM via llama.cpp.

    [llama.cpp](https://github.com/ggerganov/llama.cpp) is a project written in C/C++ for efficient inference of LLMs.
    It employs the quantized GGUF format, suitable for running these models on standard machines (even without GPUs).

    Usage example:
    ```python
    from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator
    user_message = [ChatMessage.from_user("Who is the best American actor?")]
    generator = LlamaCppGenerator(model="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)

    print(generator.run(user_message, generation_kwargs={"max_tokens": 128}))
    # {"replies": [ChatMessage(content="John Cusack", role=<ChatRole.ASSISTANT: "assistant">, name=None, meta={...}]}
    ```
    """

    def __init__(
        self,
        model: str,
        n_ctx: Optional[int] = 0,
        n_batch: Optional[int] = 512,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[List[Tool]] = None,
    ):
        """
        :param model: The path of a quantized model for text generation, for example, "zephyr-7b-beta.Q4_0.gguf".
            If the model path is also specified in the `model_kwargs`, this parameter will be ignored.
        :param n_ctx: The number of tokens in the context. When set to 0, the context will be taken from the model.
        :param n_batch: Prompt processing maximum batch size.
        :param model_kwargs: Dictionary containing keyword arguments used to initialize the LLM for text generation.
            These keyword arguments provide fine-grained control over the model loading.
            In case of duplication, these kwargs override `model`, `n_ctx`, and `n_batch` init parameters.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__).
        :param generation_kwargs:  A dictionary containing keyword arguments to customize text generation.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
        :param tools:
            A list of tools for which the model can prepare calls.
        """

        model_kwargs = model_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        if "hf_tokenizer_path" in model_kwargs:
            tokenizer = LlamaHFTokenizer.from_pretrained(model_kwargs["hf_tokenizer_path"])
            model_kwargs["tokenizer"] = tokenizer

        # check if the model_kwargs contain the essential parameters
        # otherwise, populate them with values from init parameters
        model_kwargs.setdefault("model_path", model)
        model_kwargs.setdefault("n_ctx", n_ctx)
        model_kwargs.setdefault("n_batch", n_batch)

        _check_duplicate_tool_names(tools)

        self.model_path = model
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self._model = None
        self.tools = tools

    def warm_up(self):
        if self._model is None:
            self._model = Llama(**self.model_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
              Dictionary with serialized data.
        """
        serialized_tools = [tool.to_dict() for tool in self.tools] if self.tools else None
        return default_to_dict(
            self,
            model=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            model_kwargs=self.model_kwargs,
            generation_kwargs=self.generation_kwargs,
            tools=serialized_tools,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaCppChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Run the text generation model on the given list of ChatMessages.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:  A dictionary containing keyword arguments to customize text generation.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
        :param tools:
            A list of tools for which the model can prepare calls.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        if self._model is None:
            error_msg = "The model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(error_msg)

        if not messages:
            return {"replies": []}

        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        formatted_messages = [_convert_message_to_llamacpp_format(msg) for msg in messages]

        llamacpp_tools = {}
        if tools:
            tool_definitions = [{"type": "function", "function": {**t.tool_spec}} for t in tools]
            llamacpp_tools = {"tools": tool_definitions}

        response = self._model.create_chat_completion(
            messages=formatted_messages, **updated_generation_kwargs, **llamacpp_tools
        )

        replies = []
        for choice in response["choices"]:
            chat_message = self._convert_chat_completion_choice_to_chat_message(choice, response)
            replies.append(chat_message)

        return {"replies": replies}

    @staticmethod
    def _convert_chat_completion_choice_to_chat_message(
        choice: ChatCompletionResponseChoice, response: CreateChatCompletionResponse
    ) -> ChatMessage:
        llamacpp_message = choice["message"]
        text_content = llamacpp_message.get("content", "") or None
        tool_calls = []

        if llamacpp_tool_calls := llamacpp_message.get("tool_calls", []):
            for llamacpp_tc in llamacpp_tool_calls:
                arguments_str = llamacpp_tc["function"]["arguments"]
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(
                        ToolCall(id=llamacpp_tc["id"], tool_name=llamacpp_tc["function"]["name"], arguments=arguments)
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "Llama.cpp returned a malformed JSON string for tool call arguments. This tool call "
                        "will be skipped. Tool call ID: %s, Tool name: %s, Arguments: %s",
                        llamacpp_tc["id"],
                        llamacpp_tc["function"]["name"],
                        arguments_str,
                    )

        meta = {
            "response_id": response["id"],
            "model": response["model"],
            "created": response["created"],
            "index": choice["index"],
            "finish_reason": choice["finish_reason"],
            "usage": response["usage"],
        }

        return ChatMessage.from_assistant(text=text_content, tool_calls=tool_calls, meta=meta)
