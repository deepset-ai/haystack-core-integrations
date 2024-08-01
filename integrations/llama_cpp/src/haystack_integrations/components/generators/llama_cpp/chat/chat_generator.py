import logging
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

logger = logging.getLogger(__name__)


def _convert_message_to_llamacpp_format(message: ChatMessage) -> Dict[str, str]:
    """
    Convert a message to the format expected by Llama.cpp.
    :returns: A dictionary with the following keys:
        - `role`
        - `content`
        - `name` (optional)
    """
    formatted_msg = {"role": message.role.value, "content": message.content}
    if message.name:
        formatted_msg["name"] = message.name

    return formatted_msg


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

        self.model_path = model
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self.model = None

    def warm_up(self):
        if self.model is None:
            self.model = Llama(**self.model_kwargs)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run the text generation model on the given list of ChatMessages.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:  A dictionary containing keyword arguments to customize text generation.
            For more information on the available kwargs, see
            [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        if self.model is None:
            error_msg = "The model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(error_msg)

        if not messages:
            return {"replies": []}

        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        formatted_messages = [_convert_message_to_llamacpp_format(msg) for msg in messages]

        response = self.model.create_chat_completion(messages=formatted_messages, **updated_generation_kwargs)
        replies = [
            ChatMessage(
                content=choice["message"]["content"],
                role=ChatRole[choice["message"]["role"].upper()],
                name=None,
                meta={
                    "response_id": response["id"],
                    "model": response["model"],
                    "created": response["created"],
                    "index": choice["index"],
                    "finish_reason": choice["finish_reason"],
                    "usage": response["usage"],
                },
            )
            for choice in response["choices"]
        ]

        for reply, choice in zip(replies, response["choices"]):
            tool_calls = choice.get("message", {}).get("tool_calls", [])
            if tool_calls:
                reply.meta["tool_calls"] = tool_calls
        reply.name = tool_calls[0]["function"]["name"] if tool_calls else None
        return {"replies": replies}
