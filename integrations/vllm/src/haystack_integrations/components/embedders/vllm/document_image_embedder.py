# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _encode_image_to_base64,
    _extract_image_sources_info,
    _PDFPageInfo,
)
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from more_itertools import batched
from openai import APIError, AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse
from tqdm import tqdm

from haystack_integrations.common.vllm.utils import (
    _create_async_openai_client,
    _create_openai_client,
    _prepare_image_embedding_body,
)

logger = logging.getLogger(__name__)


@component
class VLLMDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using multimodal models served with [vLLM](https://docs.vllm.ai/).

    The image or PDF page to embed is read from the file path stored in the Document's metadata.
    The embedding of each Document is stored in the `embedding` field of the Document.
    It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
    chat-style `messages` input of vLLM's OpenAI-compatible Embeddings API. Each image is sent in its own request.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with a multimodal embedding model:

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling --trust-remote-code --max-model-len 4096 \
        --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
    ```

    For the list of supported multimodal embedding models, see the
    [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.vllm import VLLMDocumentImageEmbedder

    embedder = VLLMDocumentImageEmbedder(model="TIGER-Lab/VLM2Vec-Full", instruction="Represent the given image.")

    documents = [
        Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
        Document(content="A page of a report", meta={"file_path": "report.pdf", "page_number": 1}),
    ]

    result = embedder.run(documents=documents)
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000/v1",
        file_path_meta_field: str = "file_path",
        root_path: str | None = None,
        image_size: tuple[int, int] | None = None,
        instruction: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 8,
        progress_bar: bool = True,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        raise_on_failure: bool = False,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMDocumentImageEmbedder.

        :param model: The name of the multimodal embedding model served by vLLM. Check
            [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/) for more information.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param api_base_url: The base URL of the vLLM server.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata are resolved relative to this path. If `None`, file paths are treated as absolute paths.
        :param image_size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time.
        :param instruction: Optional text sent together with each image, such as `"Represent the given image."`.
            Some models, like VLM2Vec, expect an instruction. If `None`, only the image is sent.
        :param dimensions: The number of dimensions of the resulting embedding. Only models trained with
            Matryoshka Representation Learning support this parameter.
        :param batch_size: Number of images to load into memory and process at once. In `run_async`, the requests
            for the images of a batch are sent concurrently.
        :param progress_bar: Whether to show a progress bar.
        :param timeout: Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
        :param max_retries: Maximum number of retries for failed requests. If not set, the OpenAI client
            default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param raise_on_failure: Whether to raise an exception if an embedding request fails. If `False`,
            the component logs the error and returns the Document without an embedding.
        :param extra_parameters: Additional parameters added to the body of each request to the vLLM embeddings
            endpoint, such as `truncate_prompt_tokens`. See the
            [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#openai-compatible-embeddings-api).
        """
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path
        self.image_size = image_size
        self.instruction = instruction
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs
        self.raise_on_failure = raise_on_failure
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

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list) or not all(isinstance(doc, Document) for doc in documents):
            msg = (
                "VLLMDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a single image, please use the VLLMImageEmbedder."
            )
            raise TypeError(msg)

    def _images_to_data_urls(self, documents: list[Document]) -> list[str]:
        """
        Load the image (or PDF page) of each Document and encode it as a base64 data URL.

        :param documents: Documents whose images to load.
        :returns: One data URL per Document, in the same order.
        :raises ValueError: If a PDF Document has no `page_number` in its metadata.
        :raises RuntimeError: If the image of some Documents could not be converted.
        """
        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path or ""
        )

        data_urls: list[str | None] = [None] * len(documents)
        pdf_page_infos: list[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["mime_type"] == "application/pdf":
                # `_extract_image_sources_info` already raises a ValueError if a PDF Document has no page number
                pdf_page_infos.append(
                    {
                        "doc_idx": doc_idx,
                        "path": image_source_info["path"],
                        "page_number": image_source_info["page_number"],
                    }
                )
            else:
                image_byte_stream = ByteStream.from_file_path(
                    filepath=image_source_info["path"], mime_type=image_source_info["mime_type"]
                )
                mime_type, base64_image = _encode_image_to_base64(bytestream=image_byte_stream, size=self.image_size)
                data_urls[doc_idx] = f"data:{mime_type};base64,{base64_image}"

        base64_jpeg_images_by_doc_idx = _batch_convert_pdf_pages_to_images(
            pdf_page_infos=pdf_page_infos, return_base64=True, size=self.image_size
        )
        for doc_idx, base64_jpeg_image in base64_jpeg_images_by_doc_idx.items():
            data_urls[doc_idx] = f"data:image/jpeg;base64,{base64_jpeg_image}"

        failed_doc_ids = [documents[doc_idx].id for doc_idx, url in enumerate(data_urls) if url is None]
        if failed_doc_ids:
            msg = f"Conversion failed for some documents. Document IDs: {failed_doc_ids}."
            raise RuntimeError(msg)

        return [url for url in data_urls if url is not None]

    def _prepare_body(self, image_url: str) -> dict[str, Any]:
        return _prepare_image_embedding_body(
            model=self.model,
            image_url=image_url,
            instruction=self.instruction,
            dimensions=self.dimensions,
            extra_parameters=self.extra_parameters,
        )

    def _handle_failure(self, doc: Document, exc: APIError) -> None:
        logger.exception("Failed embedding of document {doc_id} caused by {exc}", doc_id=doc.id, exc=exc)
        if self.raise_on_failure:
            raise exc

    @staticmethod
    def _update_meta(meta: dict[str, Any], response: CreateEmbeddingResponse) -> None:
        if "model" not in meta:
            meta["model"] = response.model
        if "usage" not in meta:
            meta["usage"] = dict(response.usage)
        else:
            meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
            meta["usage"]["total_tokens"] += response.usage.total_tokens

    def _build_document(self, doc: Document, embedding: list[float] | None) -> Document:
        if embedding is None:
            return replace(doc)
        new_meta = {
            **doc.meta,
            "embedding_source": {"type": "image", "file_path_meta_field": self.file_path_meta_field},
        }
        return replace(doc, meta=new_meta, embedding=embedding)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Embed a list of image Documents.

        :param documents: Documents to embed. Each Document must have the path to an image or PDF file in the
            metadata field set by `file_path_meta_field`. PDF Documents also need a `page_number` metadata field.
        :returns: A dictionary with:
            - `documents`: The input Documents with their `embedding` field populated.
            - `meta`: Information about the usage of the model.
        :raises TypeError: If the input is not a list of Documents.
        :raises ValueError: If a PDF Document has no `page_number` in its metadata.
        :raises RuntimeError: If the image of some Documents could not be converted.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        self.warm_up()
        assert self._client is not None  # noqa: S101

        new_documents: list[Document] = []
        meta: dict[str, Any] = {}
        for batch in tqdm(
            list(batched(documents, self.batch_size)),
            disable=not self.progress_bar,
            desc="Calculating image embeddings",
        ):
            data_urls = self._images_to_data_urls(list(batch))
            for doc, data_url in zip(batch, data_urls, strict=True):
                embedding = None
                try:
                    response = self._client.post(
                        "/embeddings", cast_to=CreateEmbeddingResponse, body=self._prepare_body(data_url)
                    )
                    embedding = response.data[0].embedding
                    self._update_meta(meta, response)
                except APIError as exc:
                    self._handle_failure(doc, exc)
                new_documents.append(self._build_document(doc, embedding))

        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Asynchronously embed a list of image Documents.

        The requests for the images of a batch are sent concurrently.

        :param documents: Documents to embed. Each Document must have the path to an image or PDF file in the
            metadata field set by `file_path_meta_field`. PDF Documents also need a `page_number` metadata field.
        :returns: A dictionary with:
            - `documents`: The input Documents with their `embedding` field populated.
            - `meta`: Information about the usage of the model.
        :raises TypeError: If the input is not a list of Documents.
        :raises ValueError: If a PDF Document has no `page_number` in its metadata.
        :raises RuntimeError: If the image of some Documents could not be converted.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        client = self._async_client

        new_documents: list[Document] = []
        meta: dict[str, Any] = {}
        for batch in tqdm(
            list(batched(documents, self.batch_size)),
            disable=not self.progress_bar,
            desc="Calculating image embeddings",
        ):
            data_urls = self._images_to_data_urls(list(batch))
            responses = await asyncio.gather(
                *(
                    client.post("/embeddings", cast_to=CreateEmbeddingResponse, body=self._prepare_body(url))
                    for url in data_urls
                ),
                return_exceptions=True,
            )
            for doc, response in zip(batch, responses, strict=True):
                embedding = None
                if isinstance(response, APIError):
                    self._handle_failure(doc, response)
                elif isinstance(response, BaseException):
                    raise response
                else:
                    embedding = response.data[0].embedding
                    self._update_meta(meta, response)
                new_documents.append(self._build_document(doc, embedding))

        return {"documents": new_documents, "meta": meta}
