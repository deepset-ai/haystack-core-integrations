# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils import Secret

from haystack_integrations.utils.nvidia import DEFAULT_API_URL, Model, NimBackend, is_hosted, url_validation


@component
class NvidiaChatGenerator:
    """
    Generates responses using generative chat models hosted with
    [NVIDIA NIM](https://ai.nvidia.com) on the [NVIDIA API Catalog](https://build.nvidia.com/explore/discover).

    This component uses the ChatMessage format to communicate with NVIDIA NIM models that support chat completion.

    ### Usage example

    ```python
    from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = NvidiaChatGenerator(
        model="meta/llama3-70b-instruct",
        model_arguments={
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        },
    )
    generator.warm_up()

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
        api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_arguments: Optional[Dict[str, Any]] = None,
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
        :param api_url:
            Custom API URL for the NVIDIA NIM.
        :param model_arguments:
            Additional arguments to pass to the model provider. These arguments are
            specific to a model.
            Search your model in the [NVIDIA NIM](https://ai.nvidia.com)
            to find the arguments it accepts.
        :param timeout:
            Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
            or set to 60 by default.
        """
        self._model = model
        self.api_url = url_validation(api_url)
        self._api_key = api_key
        self._model_arguments = model_arguments or {}

        self.backend: Optional[Any] = None

        self.is_hosted = is_hosted(api_url)
        if timeout is None:
            timeout = float(os.environ.get("NVIDIA_TIMEOUT", "60.0"))
        self.timeout = timeout

    def default_model(self):
        """Set default model in local NIM mode."""
        valid_models = [
            model.id for model in self.available_models if not model.base_model or model.base_model == model.id
        ]
        name = next(iter(valid_models), None)
        if name:
            warnings.warn(
                f"Default model is set as: {name}. \n"
                "Set model using model parameter. \n"
                "To get available models use available_models property.",
                UserWarning,
                stacklevel=2,
            )
            self._model = self.backend.model = name
        else:
            error_message = "No locally hosted model was found."
            raise ValueError(error_message)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.backend is not None:
            return

        self.backend = NimBackend(
            model=self._model,
            model_type="chat",
            api_url=self.api_url,
            api_key=self._api_key,
            model_kwargs=self._model_arguments,
            timeout=self.timeout,
            client=self.__class__.__name__,
        )

        if not self.is_hosted and not self._model:
            if self.backend.model:
                self.model = self.backend.model
            else:
                self.default_model()

    @classmethod
    def class_name(cls) -> str:
        return "NvidiaChatGenerator"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self._model,
            api_url=self.api_url,
            api_key=self._api_key.to_dict() if self._api_key else None,
            model_arguments=self._model_arguments,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvidiaChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        return default_from_dict(cls, data)

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with NvidiaChatGenerator.
        """
        return self.backend.models() if self.backend else []

    def _convert_messages_to_nvidia_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert a list of messages to the format expected by NVIDIA NIM API.

        :param messages: The list of ChatMessages to convert.
        :returns: A list of dictionaries in the format expected by NVIDIA NIM API.
        """
        nvidia_messages = []

        for message in messages:
            if message.is_from(ChatRole.SYSTEM):
                nvidia_messages.append({"role": "system", "content": message.text})
            elif message.is_from(ChatRole.USER):
                nvidia_messages.append({"role": "user", "content": message.text})
            elif message.is_from(ChatRole.ASSISTANT):
                nvidia_messages.append({"role": "assistant", "content": message.text})
            else:
                # Skip other message types like tool messages for now
                pass

        return nvidia_messages

    def _convert_nvidia_response_to_chat_message(self, response: Dict[str, Any]) -> ChatMessage:
        """
        Convert the response from the NVIDIA API to a ChatMessage.

        :param response: The response from the NVIDIA API.
        :returns: A ChatMessage object.
        """
        text = response.get("content", "")
        message = ChatMessage.from_assistant(text=text)

        # Add metadata to the message
        message._meta.update({
            "model": response.get("model", None),
            "finish_reason": response.get("finish_reason", None),
            "usage": response.get("usage", {}),
        })

        return message

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
    ):
        """
        Invokes the NVIDIA NIM API with the given messages and generation kwargs.

        :param messages: A list of ChatMessage instances representing the input messages.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        if self.backend is None:
            msg = "The chat model has not been loaded. Call warm_up() before running."
            raise RuntimeError(msg)

        # Convert messages to NVIDIA format
        nvidia_messages = self._convert_messages_to_nvidia_format(messages)
        
        
        # Call the backend and process response
        assert self.backend is not None
        
        responses, _ = self.backend.generate_chat(
            messages=nvidia_messages, 
        )
        
        # Convert responses to ChatMessages
        chat_messages = [self._convert_nvidia_response_to_chat_message(resp) for resp in responses]
        return {"replies": chat_messages}
