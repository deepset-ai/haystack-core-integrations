# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional

from haystack import component, default_to_dict, logging
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, serialize_callable

logger = logging.getLogger(__name__)


@component
class TogetherAIGenerator(OpenAIGenerator):
    """
    Provides an interface to generate text using an LLM running on Together AI.

    Usage example:
    ```python
    from haystack_integrations.components.generators.together_ai import TogetherAIGenerator

    generator = TogetherAIGenerator(model="deepseek-ai/DeepSeek-R1",
                                generation_kwargs={
                                "temperature": 0.9,
                                })

    print(generator.run("Who is the best Italian actor?"))
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TOGETHER_AI_API_KEY"),
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo ",
        api_base_url: Optional[str] = "https://api.together.xyz/v1",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize the TogetherAIGenerator.

        :param api_key: The Together AI API key.
        :param model: The name of the model to use.
        :param api_base_url: The base URL of the Together AI API.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param system_prompt: The system prompt to use for text generation. If not provided, the system prompt is
            omitted, and the default system prompt of the model is used.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the together.ai endpoint. See together.ai
            [documentation](https://docs.together.ai/reference/chat-completions-1) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So, 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        :param timeout:
            Timeout for together.ai Client calls, if not set it is inferred from the `TOGETHERAI_TIMEOUT` environment
            variable or set to 30.
        :param max_retries:
            Maximum retries to establish contact with together.ai if it returns an internal error, if not set it is
            inferred from the `TOGETHER_MAX_RETRIES` environment variable or set to 5.
        """

        self.api_key = api_key
        self.api_base_url = api_base_url
        self.streaming_callback = streaming_callback
        self.system_prompt = system_prompt
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        self.max_retries = max_retries

        super(TogetherAIGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
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
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
