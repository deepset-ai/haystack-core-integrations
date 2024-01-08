import logging
from typing import Any, Dict, List, Optional

from haystack import component
from llama_cpp import Llama

logger = logging.getLogger(__name__)


@component
class LlamaCppGenerator:
    """
    Generator for using a model with Llama.cpp.
    This component provides an interface to generate text using a quantized model (GGUF) using llama.cpp.

    Usage example:
    ```python
    from llama_cpp_haystack import LlamaCppGenerator
    generator = LlamaCppGenerator(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)

    print(generator.run("Who is the best American actor?", generation_kwargs={"max_tokens": 128}))
    # {'replies': ['John Cusack'], 'meta': [{"object": "text_completion", ...}]}
    ```
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: Optional[int] = 0,
        n_batch: Optional[int] = 512,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param model_path: The path of a quantized model for text generation,
            for example, "zephyr-7b-beta.Q4_0.gguf".
            If the model_path is also specified in the `model_kwargs`, this parameter will be ignored.
        :param n_ctx: The number of tokens in the context. When set to 0, the context will be taken from the model.
            If the n_ctx is also specified in the `model_kwargs`, this parameter will be ignored.
        :param n_batch: Prompt processing maximum batch size. Defaults to 512.
            If the n_batch is also specified in the `model_kwargs`, this parameter will be ignored.
        :param model_kwargs: Dictionary containing keyword arguments used to initialize the LLM for text generation.
            These keyword arguments provide fine-grained control over the model loading.
            In case of duplication, these kwargs override `model_path`, `n_ctx`, and `n_batch` init parameters.
            See Llama.cpp's [documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__)
            for more information on the available kwargs.
        :param generation_kwargs:  A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_tokens`, `temperature`, `top_k`, `top_p`,...
            See Llama.cpp's  documentation for more information:
                https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion
        """

        model_kwargs = model_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        # check if the huggingface_pipeline_kwargs contain the essential parameters
        # otherwise, populate them with values from init parameters
        model_kwargs.setdefault("model_path", model_path)
        model_kwargs.setdefault("n_ctx", n_ctx)
        model_kwargs.setdefault("n_batch", n_batch)

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self.model = None

    def warm_up(self):
        if self.model is None:
            self.model = Llama(**self.model_kwargs)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run the text generation model on the given prompt.

        :param prompt: A string representing the prompt.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_tokens`, `temperature`, `top_k`, `top_p`,...
            See Llama.cpp's  documentation for more information:
                https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion
        :return: A dictionary of the returned responses and metadata.
        """
        if self.model is None:
            error_msg = "The model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(error_msg)

        if not prompt:
            return {"replies": []}

        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        output = self.model.create_completion(prompt=prompt, **updated_generation_kwargs)
        replies = [output["choices"][0]["text"]]

        return {"replies": replies, "meta": [output]}
