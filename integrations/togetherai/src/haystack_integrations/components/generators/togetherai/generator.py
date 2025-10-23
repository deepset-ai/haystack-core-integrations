# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union, cast

from haystack import component, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.utils import Secret, serialize_callable

from .chat.chat_generator import TogetherAIChatGenerator

logger = logging.getLogger(__name__)


@component
class TogetherAIGenerator(TogetherAIChatGenerator):
    """
    Provides an interface to generate text using an LLM running on Together AI.

    Usage example:
    ```python
    from haystack_integrations.components.generators.togetherai import TogetherAIGenerator

    generator = TogetherAIGenerator(model="deepseek-ai/DeepSeek-R1",
                                generation_kwargs={
                                "temperature": 0.9,
                                })

    print(generator.run("Who is the best Italian actor?"))
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TOGETHER_API_KEY"),
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_base_url: Optional[str] = "https://api.together.xyz/v1",
        streaming_callback: Optional[StreamingCallbackT] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize the TogetherAIGenerator.

        :param api_key: The Together API key.
        :param model: The name of the model to use.
        :param api_base_url: The base URL of the Together AI API.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param system_prompt: The system prompt to use for text generation. If not provided, the system prompt is
            omitted, and the default system prompt of the model is used.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the Together AI endpoint. See Together AI
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
            Timeout for together.ai Client calls, if not set it is inferred from the `OPENAI_TIMEOUT` environment
            variable or set to 30.
        :param max_retries:
            Maximum retries to establish contact with Together AI if it returns an internal error, if not set it is
            inferred from the `OPENAI_MAX_RETRIES` environment variable or set to 5.
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
        self.system_prompt = system_prompt

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
            system_prompt=self.system_prompt,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TogetherAIGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        return cast(TogetherAIGenerator, super(TogetherAIGenerator, cls).from_dict(data))  # noqa: UP008

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(  # type: ignore[override]
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate text completions synchronously.

        :param prompt:
            The input prompt string for text generation.
        :param system_prompt:
            An optional system prompt to provide context or instructions for the generation.
            If not provided, the system prompt set in the `__init__` method will be used.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided, this will override the `streaming_callback` set in the `__init__` method.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.
        :returns:
            A dictionary with the following keys:
            - `replies`: A list of generated text completions as strings.
            - `meta`: A list of metadata dictionaries containing information about each generation,
            including model name, finish reason, and token usage statistics.
        """
        resolved_system_prompt = system_prompt or self.system_prompt
        messages = self._prepare_messages(prompt=prompt, system_prompt=resolved_system_prompt)
        # streaming_callback is verified and selected in the parent class
        chat_response = super(TogetherAIGenerator, self).run(  # noqa: UP008
            messages=messages, generation_kwargs=generation_kwargs, streaming_callback=streaming_callback
        )
        replies = chat_response["replies"]
        return self._convert_chat_response_to_generator_format(replies)

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    async def run_async(  # type: ignore[override]
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate text completions asynchronously.

        :param prompt:
            The input prompt string for text generation.
        :param system_prompt:
            An optional system prompt to provide context or instructions for the generation.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided, this will override the `streaming_callback` set in the `__init__` method.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.
        :returns:
            A dictionary with the following keys:
            - `replies`: A list of generated text completions as strings.
            - `meta`: A list of metadata dictionaries containing information about each generation,
            including model name, finish reason, and token usage statistics.
        """
        resolved_system_prompt = system_prompt or self.system_prompt
        messages = self._prepare_messages(prompt=prompt, system_prompt=resolved_system_prompt)
        # streaming_callback is verified and selected in the parent class
        chat_response = await super(TogetherAIGenerator, self).run_async(  # noqa: UP008
            messages=messages, generation_kwargs=generation_kwargs, streaming_callback=streaming_callback
        )
        replies = chat_response["replies"]
        return self._convert_chat_response_to_generator_format(replies)

    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list[ChatMessage]:
        """
        Convert prompt and system_prompt to ChatMessage format.

        :param prompt: The user prompt
        :param system_prompt: Optional system prompt
        :returns:
            List of ChatMessage objects
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage.from_system(system_prompt))
        messages.append(ChatMessage.from_user(prompt))
        return messages

    def _convert_chat_response_to_generator_format(
        self, chat_messages: list[ChatMessage]
    ) -> dict[str, Union[list[str], list[dict[str, Any]]]]:
        """
        Convert ChatGenerator response format to Generator format.

        :param chat_messages: Response from TogetherAIChatGenerator
        :returns:
            Response in Generator format with replies and meta lists
        """
        replies = []
        meta = []
        for chat_message in chat_messages:
            replies.append(chat_message.text or "")
            meta.append(chat_message.meta)
        return {"replies": replies, "meta": meta}
