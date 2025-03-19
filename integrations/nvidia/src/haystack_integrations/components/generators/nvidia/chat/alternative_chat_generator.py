# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Callable, Dict, Optional

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

from haystack_integrations.utils.nvidia import DEFAULT_API_URL


@component
class AlternativeNvidiaChatGenerator(OpenAIChatGenerator):
    """
    Generates responses using generative chat models hosted with
    [NVIDIA NIM](https://ai.nvidia.com) on the [NVIDIA API Catalog](https://build.nvidia.com/explore/discover).

    This component uses the ChatMessage format to communicate with NVIDIA NIM models that support chat completion.

    ### Usage example

    ```python
    from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = AlternativeNvidiaChatGenerator(
        model="meta/llama3-70b-instruct",
        model_arguments={
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        },
    )

    messages = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the answer to life, the universe, and everything?")
    ]
    result = generator.run(messages=messages)
    print(result["replies"])
    ```

    You need an NVIDIA API key for this component to work.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_base_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        """
        Create a NvidiaChatGenerator component.

        :param model:
            Name of the model to use for chat generation.
            See the [NVIDIA NIMs](https://ai.nvidia.com)
            for more information on the supported models.
            `Note`: If no specific model along with locally hosted API URL is provided,
            the system defaults to the available model found using /models API.
            Check supported models at [NVIDIA NIM](https://ai.nvidia.com).
        :param api_key:
            API key for the NVIDIA NIM. Set it as the `NVIDIA_API_KEY` environment
            variable or pass it here.
        :param api_base_url:
            Custom API URL for the NVIDIA NIM.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param generation_kwargs:
            Additional arguments to pass to the model provider. These arguments are
            specific to a model.
            Search your model in the [NVIDIA NIM](https://ai.nvidia.com)
            to find the arguments it accepts.
        :param timeout:
            Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
            or set to 60 by default.
        """

        if timeout is None:
            timeout = float(os.environ.get("NVIDIA_TIMEOUT", "60.0"))

        super(AlternativeNvidiaChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        # if we didn't implement the to_dict method here then the to_dict method of the superclass would be used
        # which would serialize some fields that we don't want to serialize (e.g. the ones we don't have in
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
        )
