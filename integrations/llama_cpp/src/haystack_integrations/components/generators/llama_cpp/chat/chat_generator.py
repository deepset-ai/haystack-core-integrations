import logging
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole
from llama_cpp import Llama

logger = logging.getLogger(__name__)


@component
class LlamaCppChatGenerator:
    """
    Provides an interface to generate text using LLM via llama.cpp.

    [llama.cpp](https://github.com/ggerganov/llama.cpp) is a project written in C/C++ for efficient inference of LLMs.
    It employs the quantized GGUF format, suitable for running these models on standard machines (even without GPUs).

    Usage example:
    ```python
    from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
    generator = LlamaCppGenerator(model="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)

    print(generator.run("Who is the best American actor?", generation_kwargs={"max_tokens": 128}))
    # {'replies': ['John Cusack'], 'meta': [{"object": "text_completion", ...}]}
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
        Run the text generation model on the given prompt.

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
        formatted_messages = [msg.to_openai_format() for msg in messages]

        response = self.model.create_chat_completion(messages=formatted_messages, **updated_generation_kwargs)
        replies = []
        for choice in response["choices"]:
            metadata = {
                "response_id": response["id"],
                "model": response["model"],
                "created": response["created"],
                "index": choice["index"],
                "finish_reason": choice["finish_reason"],
                "usage": response["usage"],
            }

            content = choice["message"]["content"]
            role = choice["message"]["role"].upper()

            chat_message = ChatMessage(content=content, role=ChatRole[role], name=None, meta=metadata)
            replies.append(chat_message)
        return {"replies": replies}
