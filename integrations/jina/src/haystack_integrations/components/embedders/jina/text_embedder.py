# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import httpx
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


@component
class JinaTextEmbedder:
    """
    A component for embedding strings using Jina AI models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.jina import JinaTextEmbedder

    # Make sure that the environment variable JINA_API_KEY is set

    text_embedder = JinaTextEmbedder(task="retrieval.query")

    text_to_embed = "I love pizza!"

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'jina-embeddings-v3',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),  # noqa: B008
        model: str = "jina-embeddings-v3",
        prefix: str = "",
        suffix: str = "",
        task: str | None = None,
        dimensions: int | None = None,
        late_chunking: bool | None = None,
        *,
        base_url: str = JINA_API_URL,
    ) -> None:
        """
        Create a JinaTextEmbedder component.

        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable `JINA_API_KEY` (recommended).
        :param model: The name of the Jina model to use.
            Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param task: The downstream task for which the embeddings will be used.
            The model will return the optimized embeddings for that task.
            Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
        :param dimensions: Number of desired dimension.
            Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
        :param late_chunking: A boolean to enable or disable late chunking.
            Apply the late chunking technique to leverage the model's long-context capabilities for
            generating contextual chunk embeddings.
        :param base_url: The base URL of the Jina API.

            The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.
        """

        resolved_api_key = api_key.resolve_value()

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url
        self.prefix = prefix
        self.suffix = suffix
        self._headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Accept-Encoding": "identity",
            "Content-type": "application/json",
        }
        self.task = task
        self.dimensions = dimensions
        self.late_chunking = late_chunking

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        kwargs: dict[str, Any] = {
            "api_key": self.api_key.to_dict(),
            "model": self.model_name,
            "base_url": self.base_url,
            "prefix": self.prefix,
            "suffix": self.suffix,
        }
        # Optional parameters, the following two are only supported by embeddings-v3 for now
        if self.task is not None:
            kwargs["task"] = self.task
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.late_chunking is not None:
            kwargs["late_chunking"] = self.late_chunking
        return default_to_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JinaTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _validate_input(self, text: str) -> None:
        if not isinstance(text, str):
            msg = (
                "JinaTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the JinaDocumentEmbedder."
            )
            raise TypeError(msg)

    def _prepare_request_payload(self, text: str) -> dict[str, Any]:
        text_to_embed = self.prefix + text + self.suffix

        payload: dict[str, Any] = {"input": [text_to_embed], "model": self.model_name}
        if self.task is not None:
            payload["task"] = self.task
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.late_chunking is not None:
            payload["late_chunking"] = self.late_chunking

        return payload

    @staticmethod
    def _parse_response(resp: dict[str, Any]) -> dict[str, Any]:
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        metadata = {"model": resp["model"], "usage": dict(resp["usage"].items())}
        embedding = resp["data"][0]["embedding"]

        return {"embedding": embedding, "meta": metadata}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Embed a string.

        :param text: The string to embed.
        :returns: A dictionary with following keys:
            - `embedding`: The embedding of the input string.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a string.
        """
        self._validate_input(text)
        payload = self._prepare_request_payload(text)

        with httpx.Client() as client:
            response = client.post(self.base_url, json=payload, headers=self._headers)
        resp = response.json()

        return self._parse_response(resp)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Asynchronously embed a string.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param text: The string to embed.
        :returns: A dictionary with following keys:
            - `embedding`: The embedding of the input string.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a string.
        """
        self._validate_input(text)
        payload = self._prepare_request_payload(text)

        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, json=payload, headers=self._headers)
        resp = response.json()

        return self._parse_response(resp)
