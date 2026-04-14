# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse

from haystack_integrations.common.vllm.utils import _create_openai_clients


@component
class VLLMTextEmbedder:
    """
    A component for embedding strings using models served with [vLLM](https://docs.vllm.ai/).

    It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
    OpenAI-compatible Embeddings API exposed by vLLM.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with an embedding model:

    ```bash
    vllm serve intfloat/e5-mistral-7b-instruct
    ```

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.vllm import VLLMTextEmbedder

    text_embedder = VLLMTextEmbedder(model="intfloat/e5-mistral-7b-instruct")
    print(text_embedder.run("I love pizza!"))
    ```

    ### Usage example with vLLM-specific parameters

    Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are forwarded as `extra_body`
    to the OpenAI-compatible endpoint.

    ```python
    text_embedder = VLLMTextEmbedder(
        model="jinaai/jina-embeddings-v3",
        extra_parameters={"dimensions": 32, "truncate_prompt_tokens": 256},
    )
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000/v1",
        prefix: str = "",
        suffix: str = "",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMTextEmbedder.

        :param model: The name of the model served by vLLM (e.g., "intfloat/e5-mistral-7b-instruct").
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param api_base_url: The base URL of the vLLM server.
        :param prefix: A string to add at the beginning of each text to embed.
        :param suffix: A string to add at the end of each text to embed.
        :param timeout: Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
        :param max_retries: Maximum number of retries for failed requests. If not set, the OpenAI client
            default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param extra_parameters: Additional parameters forwarded as `extra_body` to the vLLM embeddings
            endpoint. Use this to pass parameters not part of the standard OpenAI Embeddings API, such as
            `dimensions` (for Matryoshka models), `truncate_prompt_tokens`, `truncation_side`,
            `additional_data`, `use_activation`, etc. See the
            [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models.html#openai-compatible-embeddings-api).
        """
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.prefix = prefix
        self.suffix = suffix
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs
        self.extra_parameters = extra_parameters

        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """Create the OpenAI clients."""
        if self._is_warmed_up:
            return
        self._client, self._async_client = _create_openai_clients(
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
        self._is_warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict() if self.api_key else None,
            api_base_url=self.api_base_url,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
            extra_parameters=self.extra_parameters,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLLMTextEmbedder":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    def _prepare_input(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            msg = (
                "VLLMTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the VLLMDocumentEmbedder."
            )
            raise TypeError(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self.prefix + text + self.suffix,
            "encoding_format": "float",
        }
        if self.extra_parameters:
            kwargs["extra_body"] = self.extra_parameters
        return kwargs

    @staticmethod
    def _prepare_output(response: CreateEmbeddingResponse) -> dict[str, Any]:
        return {
            "embedding": response.data[0].embedding,
            "meta": {"model": response.model, "usage": dict(response.usage)},
        }

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Embed a single string.

        :param text: Text to embed.
        :returns: A dictionary with:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        kwargs = self._prepare_input(text)
        if not self._is_warmed_up:
            self.warm_up()
        assert self._client is not None  # noqa: S101
        response = self._client.embeddings.create(**kwargs)
        return self._prepare_output(response)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Asynchronously embed a single string.

        :param text: Text to embed.
        :returns: A dictionary with:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        kwargs = self._prepare_input(text)
        if not self._is_warmed_up:
            self.warm_up()
        assert self._async_client is not None  # noqa: S101
        response = await self._async_client.embeddings.create(**kwargs)
        return self._prepare_output(response)
