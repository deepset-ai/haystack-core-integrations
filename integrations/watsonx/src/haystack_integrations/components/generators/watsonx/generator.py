# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from haystack import component, logging
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.utils import Secret

from .chat.chat_generator import WatsonxChatGenerator

logger = logging.getLogger(__name__)


@component
class WatsonxGenerator(WatsonxChatGenerator):
    """
    Enables text completions using IBM's watsonx.ai foundation models.

    This component extends WatsonxChatGenerator to provide the standard Generator interface that works with prompt
    strings instead of ChatMessage objects.

    The generator works with IBM's foundation models that are listed
    [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&audience=wdp).

    You can customize the generation behavior by passing parameters to the watsonx.ai API through the
    `generation_kwargs` argument. These parameters are passed directly to the watsonx.ai inference endpoint.

    For details on watsonx.ai API parameters, see
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.watsonx.generator import WatsonxGenerator
    from haystack.utils import Secret

    generator = WatsonxGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        model="ibm/granite-13b-chat-v2",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )

    response = generator.run(
        prompt="Explain quantum computing in simple terms",
        system_prompt="You are a helpful physics teacher.",
    )
    print(response)
    ```
    Output:
    ```
    {
        "replies": ["Quantum computing uses quantum-mechanical phenomena like...."],
        "meta": [
            {
                "model": "ibm/granite-13b-chat-v2",
                "project_id": "your-project-id",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 45,
                    "total_tokens": 57,
                },
            }
        ],
    }
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),  # noqa: B008
        model: str = "ibm/granite-3-3-8b-instruct",
        project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),  # noqa: B008
        api_base_url: str = "https://us-south.ml.cloud.ibm.com",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        verify: bool | str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> None:
        """
        Creates an instance of WatsonxGenerator.

        Before initializing the component, you can set environment variables:
        - `WATSONX_TIMEOUT` to override the default timeout
        - `WATSONX_MAX_RETRIES` to override the default retry count

        :param api_key: IBM Cloud API key for watsonx.ai access.
            Can be set via `WATSONX_API_KEY` environment variable or passed directly.
        :param model: The model ID to use for completions. Defaults to "ibm/granite-13b-chat-v2".
            Available models can be found in your IBM Cloud account.
        :param project_id: IBM Cloud project ID
        :param api_base_url: Custom base URL for the API endpoint.
            Defaults to "https://us-south.ml.cloud.ibm.com".
        :param system_prompt: The system prompt to use for text generation.
        :param generation_kwargs: Additional parameters to control text generation.
            These parameters are passed directly to the watsonx.ai inference endpoint.
            Supported parameters include:
            - `temperature`: Controls randomness (lower = more deterministic)
            - `max_new_tokens`: Maximum number of tokens to generate
            - `min_new_tokens`: Minimum number of tokens to generate
            - `top_p`: Nucleus sampling probability threshold
            - `top_k`: Number of highest probability tokens to consider
            - `repetition_penalty`: Penalty for repeated tokens
            - `length_penalty`: Penalty based on output length
            - `stop_sequences`: List of sequences where generation should stop
            - `random_seed`: Seed for reproducible results
        :param timeout: Timeout in seconds for API requests.
            Defaults to environment variable `WATSONX_TIMEOUT` or 30 seconds.
        :param max_retries: Maximum number of retry attempts for failed requests.
            Defaults to environment variable `WATSONX_MAX_RETRIES` or 5.
        :param verify: SSL verification setting. Can be:
            - True: Verify SSL certificates (default)
            - False: Skip verification (insecure)
            - Path to CA bundle for custom certificates
        :param streaming_callback: A callback function for streaming responses.
        """
        super(WatsonxGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            project_id=project_id,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            verify=verify,
            streaming_callback=streaming_callback,
        )
        self.system_prompt = system_prompt

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        data = super(WatsonxGenerator, self).to_dict()  # noqa: UP008
        data["init_parameters"]["system_prompt"] = self.system_prompt
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatsonxGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        return cast(WatsonxGenerator, super(WatsonxGenerator, cls).from_dict(data))  # noqa: UP008

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(  # type: ignore[override]
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
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
        chat_response = super(WatsonxGenerator, self).run(  # noqa: UP008
            messages=messages, generation_kwargs=generation_kwargs, streaming_callback=streaming_callback
        )
        replies = chat_response["replies"]
        return self._convert_chat_response_to_generator_format(replies)

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    async def run_async(  # type: ignore[override]
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
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
        chat_response = await super(WatsonxGenerator, self).run_async(  # noqa: UP008
            messages=messages, generation_kwargs=generation_kwargs, streaming_callback=streaming_callback
        )
        replies = chat_response["replies"]
        return self._convert_chat_response_to_generator_format(replies)

    def _prepare_messages(self, prompt: str, system_prompt: str | None = None) -> list[ChatMessage]:
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
    ) -> dict[str, list[str] | list[dict[str, Any]]]:
        """
        Convert ChatGenerator response format to Generator format.

        :param chat_messages: Response from WatsonxChatGenerator
        :returns:
            Response in Generator format with replies and meta lists
        """
        replies = []
        meta = []
        for chat_message in chat_messages:
            replies.append(chat_message.text or "")
            meta.append(chat_message.meta)
        return {"replies": replies, "meta": meta}
