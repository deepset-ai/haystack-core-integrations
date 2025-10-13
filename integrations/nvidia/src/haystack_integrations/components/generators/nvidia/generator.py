# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from haystack_integrations.utils.nvidia import DEFAULT_API_URL, Client, Model, NimBackend, is_hosted, url_validation


@component
class NvidiaGenerator:
    """
    Generates text using generative models hosted with
    [NVIDIA NIM](https://ai.nvidia.com) on the [NVIDIA API Catalog](https://build.nvidia.com/explore/discover).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.nvidia import NvidiaGenerator

    generator = NvidiaGenerator(
        model="meta/llama3-8b-instruct",
        model_arguments={
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        },
    )
    generator.warm_up()

    result = generator.run(prompt="What is the answer?")
    print(result["replies"])
    print(result["meta"])
    print(result["usage"])
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
        Create a NvidiaGenerator component.

        :param model:
            Name of the model to use for text generation.
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

    @classmethod
    def class_name(cls) -> str:
        return "NvidiaGenerator"

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
            self._model = name
            if self.backend:
                self.backend.model = name
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
            client=Client.NVIDIA_GENERATOR,
        )

        if not self.is_hosted and not self._model:
            if self.backend.model:
                self.model = self.backend.model
            else:
                self.default_model()

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

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with ChatNVIDIA.
        """
        return self.backend.models() if self.backend else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvidiaGenerator":
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

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]:
        """
        Queries the model with the provided prompt.

        :param prompt:
            Text to be sent to the generative model.
        :returns:
            A dictionary with the following keys:
            - `replies` - Replies generated by the model.
            - `meta` - Metadata for each reply.
        """
        if self.backend is None:
            msg = "The generation model has not been loaded. Call warm_up() before running."
            raise RuntimeError(msg)

        assert self.backend is not None
        replies, meta = self.backend.generate(prompt=prompt)

        return {"replies": replies, "meta": meta}
