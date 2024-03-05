# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Type

import requests
from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack_integrations.components.generators.nvidia.providers import ModelProvider, NvidiaProvider

REQUESTS_TIMEOUT = 30

FUNCTIONS_ENDPOINT = "https://api.nvcf.nvidia.com/v2/nvcf/functions"

MODEL_PROVIDERS: Dict[str, Type[ModelProvider]] = {
    "playground_nv_llama2_rlhf_70b": NvidiaProvider,
    "playground_steerlm_llama_70b": NvidiaProvider,
    "playground_nemotron_steerlm_8b": NvidiaProvider,
    "playground_nemotron_qa_8b": NvidiaProvider,
}


@component
class NvidiaGenerator:
    """
    TODO
    """

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("NVIDIA_API_KEY"),
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
        bad: Optional[List[str]] = None,
        stop: Optional[List[str]] = None,
    ):
        """
        Create a NvidiaGenerator component.

        :param model:
            Name of the model to use for text generation.
            See the [Nvidia catalog](https://catalog.ngc.nvidia.com/ai-foundation-models) to know the  supported models.
        :param api_key:
            Nvidia API key to use for authentication.
        :param temperature:
            The sampling temperature to use for text generation. The higher the temperature value is, the less
            deterministic the output text will be.
            It is not recommended to modify both `temperature` and `top_p` in the same call.
        :param top_p:
            The top-p sampling mass used for text generation.
            The top-p value determines the probability mass that is sampled at sampling time.
            For example, if `top_p` = 0.2, only the most likely tokens (summing to 0.2 cumulative probability)
            will be sampled.
            It is not recommended to modify both `temperature` and `top_p` in the same call.
        :param max_tokens:
            The maximum number of tokens to generate in any given call.
            Note that the model is not aware of this value, and generation will simply stop at the number of tokens
            specified.
        :param seed:
            If specified, the system will make a best effort to sample deterministically, such that repeated requests
            with the same seed and parameters should return the same result.
        :param bad:
            A  of words not to use. The words are case sensitive
        :param stop:
            A list of strings where the API will stop generating further tokens.
            The returned text will not contain the stop sequence.

        :raises ValueError: If `model` is not supported by Nvidia.
        """
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._seed = seed
        self._bad = bad
        self._stop = stop

        # We use a Session to make requests as it's a bit faster as it reuses the same connection
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._api_key.resolve_value()}",
                "Content-Type": "application/json",
            }
        )

        self._headers = {
            "Authorization": f"Bearer {self._api_key.resolve_value()}",
            "Content-Type": "application/json",
        }

        if self._model not in MODEL_PROVIDERS:
            models = ", ".join(MODEL_PROVIDERS)
            msg = f"Model {self._model} is not supported, available models are: {models}"
            raise ValueError(msg)

        model_info = self._model_info()
        provider = MODEL_PROVIDERS[self._model]
        self._model_provider = provider(
            session=self._session,
            model_id=model_info["id"],
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            seed=self._seed,
            bad=self._bad,
            stop=self._stop,
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
            api_key=self._api_key.to_dict(),
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            seed=self._seed,
            bad=self._bad,
            stop=self._stop,
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

    def _model_info(self) -> Dict[str, Any]:
        res = self._session.get(url=FUNCTIONS_ENDPOINT, headers=self._headers, timeout=REQUESTS_TIMEOUT)
        res.raise_for_status()
        for model in res.json()["functions"]:
            if model["name"] == self._model:
                return model
        msg = f"Model {self._model} is not supported"
        raise ValueError(msg)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str):
        return self._model_provider.send(messages=[{"role": "user", "content": prompt}])
