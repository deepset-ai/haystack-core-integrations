# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack_integrations.utils.nvidia import NimBackend, is_hosted, url_validation

_DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1"


@component
class NvidiaGenerator:
    """
    Generates text using generative models hosted with
    [NVIDIA NIM](https://ai.nvidia.com) on on the [NVIDIA API Catalog](https://build.nvidia.com/explore/discover).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.nvidia import NvidiaGenerator

    generator = NvidiaGenerator(
        model="meta/llama3-70b-instruct",
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
        api_url: str = _DEFAULT_API_URL,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_arguments: Optional[Dict[str, Any]] = None,
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
        """
        self._model = model
        self._api_url = url_validation(api_url, _DEFAULT_API_URL, ["v1/chat/completions"])
        self._api_key = api_key
        self._model_arguments = model_arguments or {}

        self._backend: Optional[Any] = None

        self.is_hosted = is_hosted(api_url)

    def default_model(self):
        """Set default model in local NIM mode."""
        valid_models = [
            model.id for model in self._backend.models() if not model.base_model or model.base_model == model.id
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
            self._model = self._backend.model_name = name
        else:
            error_message = "No locally hosted model was found."
            raise ValueError(error_message)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._backend is not None:
            return

        if self._api_url == _DEFAULT_API_URL and self._api_key is None:
            msg = "API key is required for hosted NVIDIA NIMs."
            raise ValueError(msg)
        self._backend = NimBackend(
            self._model,
            api_url=self._api_url,
            api_key=self._api_key,
            model_kwargs=self._model_arguments,
        )

        if not self.is_hosted and not self._model:
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
            api_url=self._api_url,
            api_key=self._api_key.to_dict() if self._api_key else None,
            model_arguments=self._model_arguments,
        )

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
    def run(self, prompt: str):
        """
        Queries the model with the provided prompt.

        :param prompt:
            Text to be sent to the generative model.
        :returns:
            A dictionary with the following keys:
            - `replies` - Replies generated by the model.
            - `meta` - Metadata for each reply.
        """
        if self._backend is None:
            msg = "The generation model has not been loaded. Call warm_up() before running."
            raise RuntimeError(msg)

        assert self._backend is not None
        replies, meta = self._backend.generate(prompt=prompt)

        return {"replies": replies, "meta": meta}
