# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from ._nim_backend import NimBackend
from ._nvcf_backend import NvcfBackend
from .backend import GeneratorBackend


@component
class NvidiaGenerator:
    """
    A component for generating text using generative models provided by
    [NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
    and NVIDIA Inference Microservices.

    Usage example:
    ```python
    from haystack_integrations.components.generators.nvidia import NvidiaGenerator

    generator = NvidiaGenerator(
        model="nv_llama2_rlhf_70b",
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
    """

    def __init__(
        self,
        model: str,
        api_url: Optional[str] = None,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_arguments: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a NvidiaGenerator component.

        :param model:
            Name of the model to use for text generation.
            See the [Nvidia catalog](https://catalog.ngc.nvidia.com/ai-foundation-models)
            for more information on the supported models.
        :param api_key:
            API key for the NVIDIA AI Foundation Endpoints.
        :param api_url:
            Custom API URL for the NVIDIA Inference Microservices.
        :param model_arguments:
            Additional arguments to pass to the model provider. Different models accept different arguments.
            Search your model in the [Nvidia catalog](https://catalog.ngc.nvidia.com/ai-foundation-models)
            to know the supported arguments.
        """
        self._model = model
        self._api_url = api_url
        self._api_key = api_key
        self._model_arguments = model_arguments or {}

        self._backend: Optional[GeneratorBackend] = None

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._backend is not None:
            return

        if self._api_url is None:
            if self._api_key is None:
                msg = "API key is required for NVIDIA AI Foundation Endpoints."
                raise ValueError(msg)
            self._backend = NvcfBackend(self._model, api_key=self._api_key, model_kwargs=self._model_arguments)
        else:
            self._backend = NimBackend(
                self._model,
                api_url=self._api_url,
                api_key=self._api_key,
                model_kwargs=self._model_arguments,
            )

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
