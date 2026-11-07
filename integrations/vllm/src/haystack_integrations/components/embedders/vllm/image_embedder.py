# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component
from haystack.dataclasses import ImageContent
from haystack.utils import Secret
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse

from haystack_integrations.common.vllm.utils import (
    _create_async_openai_client,
    _create_openai_client,
    _prepare_image_embedding_body,
)


@component
class VLLMImageEmbedder:
    """
    A component for embedding a single image using multimodal models served with [vLLM](https://docs.vllm.ai/).

    Use it to embed an image query, for example to find Documents similar to an image.
    To embed image Documents, use the `VLLMDocumentImageEmbedder`.
    It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
    chat-style `messages` input of vLLM's OpenAI-compatible Embeddings API.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with a multimodal embedding model:

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling --trust-remote-code --max-model-len 4096 \
        --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
    ```

    ### Usage example

    ```python
    from haystack.dataclasses import ImageContent
    from haystack_integrations.components.embedders.vllm import VLLMImageEmbedder

    image = ImageContent.from_file_path("cat.jpg")

    image_embedder = VLLMImageEmbedder(model="TIGER-Lab/VLM2Vec-Full", instruction="Represent the given image.")

    result = image_embedder.run(image=image)
    print(result["embedding"])

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000/v1",
        instruction: str | None = None,
        dimensions: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMImageEmbedder.

        :param model: The name of the multimodal embedding model served by vLLM. Check
            [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/) for more information.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param api_base_url: The base URL of the vLLM server.
        :param instruction: Optional text sent together with the image, such as `"Represent the given image."`.
            Some models, like VLM2Vec, expect an instruction. If `None`, only the image is sent.
        :param dimensions: The number of dimensions of the resulting embedding. Only models trained with
            Matryoshka Representation Learning support this parameter.
        :param timeout: Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
        :param max_retries: Maximum number of retries for failed requests. If not set, the OpenAI client
            default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param extra_parameters: Additional parameters added to the body of the request to the vLLM embeddings
            endpoint, such as `truncate_prompt_tokens`. See the
            [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#openai-compatible-embeddings-api).
        """
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.instruction = instruction
        self.dimensions = dimensions
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

    def _prepare_body(self, image: ImageContent) -> dict[str, Any]:
        if not isinstance(image, ImageContent):
            msg = (
                "VLLMImageEmbedder expects an ImageContent as input. "
                "In case you want to embed a list of image Documents, please use the VLLMDocumentImageEmbedder."
            )
            raise TypeError(msg)

        mime_type = image.mime_type or "image/jpeg"
        return _prepare_image_embedding_body(
            model=self.model,
            image_url=f"data:{mime_type};base64,{image.base64_image}",
            instruction=self.instruction,
            dimensions=self.dimensions,
            extra_parameters=self.extra_parameters,
        )

    @staticmethod
    def _prepare_output(response: CreateEmbeddingResponse) -> dict[str, list[float] | dict[str, Any]]:
        return {
            "embedding": response.data[0].embedding,
            "meta": {"model": response.model, "usage": dict(response.usage)},
        }

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, image: ImageContent) -> dict[str, list[float] | dict[str, Any]]:
        """
        Embed a single image.

        :param image: The image to embed.
        :returns: A dictionary with:
            - `embedding`: The embedding of the input image.
            - `meta`: Information about the usage of the model.
        :raises TypeError: If the input is not an `ImageContent`.
        """
        body = self._prepare_body(image)
        self.warm_up()
        assert self._client is not None  # noqa: S101
        response = self._client.post("/embeddings", cast_to=CreateEmbeddingResponse, body=body)
        return self._prepare_output(response)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, image: ImageContent) -> dict[str, list[float] | dict[str, Any]]:
        """
        Asynchronously embed a single image.

        :param image: The image to embed.
        :returns: A dictionary with:
            - `embedding`: The embedding of the input image.
            - `meta`: Information about the usage of the model.
        :raises TypeError: If the input is not an `ImageContent`.
        """
        body = self._prepare_body(image)
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        response = await self._async_client.post("/embeddings", cast_to=CreateEmbeddingResponse, body=body)
        return self._prepare_output(response)
