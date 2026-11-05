# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse

from haystack_integrations.common.vllm.utils import (
    _create_async_openai_client,
    _create_openai_client,
    _to_sparse_embedding,
)


@component
class VLLMSparseTextEmbedder:
    """
    A component for computing sparse embeddings of strings using models served with [vLLM](https://docs.vllm.ai/).

    It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
    OpenAI-compatible Embeddings API exposed by vLLM.
    The model must be a SPLADE model served with vLLM's `BertSpladeSparseEmbeddingModel` architecture, which returns
    one weight per vocabulary token. The component keeps the nonzero weights as a `SparseEmbedding`.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with a SPLADE model:

    ```bash
    vllm serve prithivida/Splade_PP_en_v1 --runner pooling \\
        --hf-overrides '{"architectures": ["BertSpladeSparseEmbeddingModel"]}'
    ```

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.vllm import VLLMSparseTextEmbedder

    text_embedder = VLLMSparseTextEmbedder(model="prithivida/Splade_PP_en_v1")
    print(text_embedder.run("I love pizza!"))

    # {'sparse_embedding': SparseEmbedding(indices=[...], values=[...]), 'meta': {...}}
    ```

    ### Usage example with vLLM-specific parameters

    Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are forwarded as `extra_body`
    to the OpenAI-compatible endpoint.

    ```python
    text_embedder = VLLMSparseTextEmbedder(
        model="prithivida/Splade_PP_en_v1",
        extra_parameters={"truncate_prompt_tokens": 256, "truncation_side": "right"},
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
        Creates an instance of VLLMSparseTextEmbedder.

        :param model: The name of the SPLADE model served by vLLM (e.g., "prithivida/Splade_PP_en_v1").
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
            `truncate_prompt_tokens` or `truncation_side`. See the
            [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#openai-compatible-embeddings-api).
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

    def warm_up(self) -> None:
        """Create the synchronous OpenAI client."""
        if self._client is None:
            self._client = _create_openai_client(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client_kwargs=self.http_client_kwargs,
            )

    async def warm_up_async(self) -> None:
        """Create the asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = _create_async_openai_client(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client_kwargs=self.http_client_kwargs,
            )

    def close(self) -> None:
        """Close the synchronous OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the asynchronous OpenAI client."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def _prepare_input(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            msg = (
                "VLLMSparseTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the VLLMSparseDocumentEmbedder."
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
    def _prepare_output(response: CreateEmbeddingResponse) -> dict[str, SparseEmbedding | dict[str, Any]]:
        return {
            "sparse_embedding": _to_sparse_embedding(response.data[0].embedding),
            "meta": {"model": response.model, "usage": dict(response.usage)},
        }

    @component.output_types(sparse_embedding=SparseEmbedding, meta=dict[str, Any])
    def run(self, text: str) -> dict[str, SparseEmbedding | dict[str, Any]]:
        """
        Embed a single string into a sparse vector.

        :param text: Text to embed.
        :returns: A dictionary with:
            - `sparse_embedding`: The sparse embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        kwargs = self._prepare_input(text)
        self.warm_up()
        assert self._client is not None  # noqa: S101
        response = self._client.embeddings.create(**kwargs)
        return self._prepare_output(response)

    @component.output_types(sparse_embedding=SparseEmbedding, meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, SparseEmbedding | dict[str, Any]]:
        """
        Asynchronously embed a single string into a sparse vector.

        :param text: Text to embed.
        :returns: A dictionary with:
            - `sparse_embedding`: The sparse embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        kwargs = self._prepare_input(text)
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        response = await self._async_client.embeddings.create(**kwargs)
        return self._prepare_output(response)
