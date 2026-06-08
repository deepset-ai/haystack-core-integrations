# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from google.genai.types import Content, EmbedContentConfig, Part
from haystack import Document, component, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _encode_image_to_base64,
    _PDFPageInfo,
)
from haystack.dataclasses import ByteStream
from haystack.utils.auth import Secret
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing_extensions import NotRequired, TypedDict

from haystack_integrations.components.common.google_genai.utils import _get_client

logger = logging.getLogger(__name__)

IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "application/pdf"]


class _SourceInfo(TypedDict):
    path: Path
    mime_type: str
    send_raw: bool
    page_number: NotRequired[int]


SUPPORTED_MIME_TYPES = [
    *IMAGE_MIME_TYPES,
    "video/mp4",
    "video/quicktime",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
]


def _extract_sources_info(documents: list[Document], file_path_meta_field: str, root_path: str) -> list[_SourceInfo]:
    """
    Extracts the source information from the documents.

    Adapted from haystack.components.converters.image.image_utils._extract_image_sources_info to accept non-image files.

    :param documents: List of documents to extract source information from.
    :param file_path_meta_field: The metadata field in the Document that contains the file path.
    :param root_path: The root directory path where document files are located.

    :returns:
        A list of _SourceInfo dictionaries, each containing the path and type of the file.
        If the file is a PDF and a page number is provided, the dictionary also contains the page number.
        Files that are not images, or PDFs without a page number, have `send_raw` set to True,
        meaning they will be sent as raw bytes without image processing.
    :raises ValueError: If the document is missing the file_path_meta_field key in its metadata or the file path is
        invalid.
    """
    sources_info: list[_SourceInfo] = []
    for doc in documents:
        file_path = doc.meta.get(file_path_meta_field)
        if file_path is None:
            err_msg = (
                f"Document with ID '{doc.id}' is missing the '{file_path_meta_field}' key in its metadata."
                f" Please ensure that the documents you are trying to convert have this key set."
            )
            raise ValueError(err_msg)

        resolved_file_path = Path(root_path, file_path)
        if not resolved_file_path.is_file():
            err_msg = (
                f"Document with ID '{doc.id}' has an invalid file path '{resolved_file_path}'. "
                "Please ensure that the documents you are trying to convert have valid file paths."
            )
            raise ValueError(err_msg)

        mime_type = doc.meta.get("mime_type") or mimetypes.guess_type(resolved_file_path)[0]

        if mime_type not in SUPPORTED_MIME_TYPES:
            err_msg = (
                f"Document with ID '{doc.id}' has an unsupported MIME type '{mime_type}'. "
                f"Supported MIME types are: {', '.join(SUPPORTED_MIME_TYPES)}"
            )
            raise ValueError(err_msg)

        send_raw = mime_type not in IMAGE_MIME_TYPES
        source_info: _SourceInfo = {"path": resolved_file_path, "mime_type": mime_type, "send_raw": send_raw}

        # If mimetype is PDF, we use the page number to convert the right page to an image. If not provided,
        # we send the raw PDF bytes without converting to an image.
        if mime_type == "application/pdf":
            page_number = doc.meta.get("page_number")
            if page_number is None:
                source_info["send_raw"] = True
            else:
                source_info["page_number"] = page_number

        sources_info.append(source_info)

    return sources_info


@component
class GoogleGenAIMultimodalDocumentEmbedder:
    """
    Computes non-textual document embeddings using Google AI models.

    It supports images, PDFs, video and audio files. They are mapped to vectors in a single vector space.

    To embed textual documents, use the GoogleGenAIDocumentEmbedder.
    To embed a string, like a user query, use the GoogleGenAITextEmbedder.

    ### Authentication examples

    **1. Gemini Developer API (API Key Authentication)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIMultimodalDocumentEmbedder

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    document_embedder = GoogleGenAIMultimodalDocumentEmbedder(model="gemini-embedding-2-preview")

    **2. Vertex AI (Application Default Credentials)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIMultimodalDocumentEmbedder

    # Using Application Default Credentials (requires gcloud auth setup)
    document_embedder = GoogleGenAIMultimodalDocumentEmbedder(
        api="vertex",
        vertex_ai_project="my-project",
        vertex_ai_location="us-central1",
        model="gemini-embedding-2-preview"
    )
    ```

    **3. Vertex AI (API Key Authentication)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIMultimodalDocumentEmbedder

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    document_embedder = GoogleGenAIMultimodalDocumentEmbedder(
        api="vertex",
        model="gemini-embedding-2-preview"
    )
    ```

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIMultimodalDocumentEmbedder

    doc = Document(content=None, meta={"file_path": "path/to/image.jpg"})

    document_embedder = GoogleGenAIMultimodalDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: str | None = None,
        vertex_ai_location: str | None = None,
        file_path_meta_field: str = "file_path",
        root_path: str | None = None,
        image_size: tuple[int, int] | None = None,
        model: str = "gemini-embedding-2",
        batch_size: int = 6,
        progress_bar: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an GoogleGenAIMultimodalDocumentEmbedder component.

        :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
            Not needed if using Vertex AI with Application Default Credentials.
            Go to https://aistudio.google.com/app/apikey for a Gemini API key.
            Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
        :param api: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
        :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
            Application Default Credentials.
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
            Required when using Vertex AI with Application Default Credentials.
        :param file_path_meta_field:
            The metadata field in the Document that contains the file path to the file to embed.
        :param root_path:
            The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param image_size:
            Only used for images and PDF pages. If provided, resizes the image to fit within the specified dimensions
            (width, height) while maintaining aspect ratio. This reduces file size, memory usage, and processing time,
            which is beneficial when working with models that have resolution constraints or when transmitting images
            to remote services.
        :param model:
            The name of the model to use for calculating embeddings.
        :param batch_size:
            Number of documents to embed at once. Maximum batch size varies depending on the input type.
            See [Google AI documentation](https://ai.google.dev/gemini-api/docs/embeddings#supported-modalities) for
            more information.
        :param progress_bar:
            If `True`, shows a progress bar when running.
        :param config:
            A dictionary of keyword arguments to configure embedding content configuration.
            You can for example set the output dimensionality of the embedding: `{"output_dimensionality": 768}`.
            See [Google API documentation](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig)
            for the available options.
        """
        self._api_key = api_key
        self._api = api
        self._vertex_ai_project = vertex_ai_project
        self._vertex_ai_location = vertex_ai_location
        self._model = model
        self._file_path_meta_field = file_path_meta_field
        self._root_path = root_path or ""
        self._image_size = image_size
        self._batch_size = batch_size
        self._progress_bar = progress_bar
        self._config = config

        self._client = _get_client(
            api_key=api_key,
            api=api,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
        )

    def _extract_parts_to_embed(self, documents: list[Document]) -> list[Part]:
        """
        Validates the input documents and extracts the files to embed in the format expected by the Google AI API.

        :param documents:
            Documents to embed.

        :returns:
            List of files to embed in the format expected by the Google AI API.

        :raises TypeError:
            If the input is not a list of `Documents`.
        :raises RuntimeError:
            If the conversion of some documents fails.
        """
        if not isinstance(documents, list) or not all(isinstance(d, Document) for d in documents):
            msg = (
                "GoogleGenAIMultimodalDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the GoogleGenAITextEmbedder."
            )
            raise TypeError(msg)

        parts_source_info = _extract_sources_info(
            documents=documents, file_path_meta_field=self._file_path_meta_field, root_path=self._root_path
        )

        parts_to_embed: list[Part | None] = [None] * len(documents)
        pdf_page_infos: list[_PDFPageInfo] = []

        for doc_idx, part_source_info in enumerate(parts_source_info):
            if part_source_info.get("send_raw", False):
                parts_to_embed[doc_idx] = Part.from_bytes(
                    data=part_source_info["path"].read_bytes(), mime_type=part_source_info["mime_type"]
                )
            elif part_source_info["mime_type"] == "application/pdf" and part_source_info.get("page_number") is not None:
                # Store PDF documents for later processing
                pdf_page_info: _PDFPageInfo = {
                    "doc_idx": doc_idx,
                    "path": part_source_info["path"],
                    "page_number": part_source_info["page_number"],
                }
                pdf_page_infos.append(pdf_page_info)
            else:
                # Process images directly
                image_byte_stream = ByteStream.from_file_path(
                    filepath=part_source_info["path"], mime_type=part_source_info["mime_type"]
                )
                resolved_mime_type, base64_image = _encode_image_to_base64(
                    bytestream=image_byte_stream, size=self._image_size
                )
                parts_to_embed[doc_idx] = Part.from_bytes(
                    data=base64.b64decode(base64_image),
                    mime_type=resolved_mime_type or part_source_info["mime_type"],
                )

        base64_jpeg_images_by_doc_idx = _batch_convert_pdf_pages_to_images(
            pdf_page_infos=pdf_page_infos, return_base64=True, size=self._image_size
        )
        for doc_idx, base64_jpeg_image in base64_jpeg_images_by_doc_idx.items():
            parts_to_embed[doc_idx] = Part.from_bytes(
                data=base64.b64decode(str(base64_jpeg_image)), mime_type="image/jpeg"
            )

        none_parts_doc_ids = [documents[doc_idx].id for doc_idx, part in enumerate(parts_to_embed) if part is None]
        if none_parts_doc_ids:
            msg = f"Conversion failed for some documents. Document IDs: {none_parts_doc_ids}."
            raise RuntimeError(msg)

        # tested above that part is not None, but mypy doesn't know that
        return parts_to_embed  # type: ignore[return-value]

    def _embed_batch(
        self, parts_to_embed: list[Part], batch_size: int
    ) -> tuple[list[list[float] | None], dict[str, Any]]:
        """
        Embed a list of parts in batches.
        """
        resolved_config = EmbedContentConfig(**self._config) if self._config else None

        all_embeddings: list[list[float] | None] = []
        meta: dict[str, Any] = {}
        for i in tqdm(
            range(0, len(parts_to_embed), batch_size), disable=not self._progress_bar, desc="Calculating embeddings"
        ):
            batch = parts_to_embed[i : i + batch_size]
            args: dict[str, Any] = {"model": self._model, "contents": [Content(parts=[p]) for p in batch]}
            if resolved_config:
                args["config"] = resolved_config

            response = self._client.models.embed_content(**args)

            embeddings: list[list[float] | None] = []
            if response.embeddings:
                for el in response.embeddings:
                    if el.values:
                        embeddings.append(el.values)
                    else:
                        logger.warning("An embedding in the API response has no values, setting it to None.")
                        embeddings.append(None)
                all_embeddings.extend(embeddings)
            else:
                logger.warning(
                    "No embeddings returned by the API for a batch of {count} document(s). "
                    "Their embeddings will be set to None.",
                    count=len(batch),
                )
                all_embeddings.extend([None] * len(batch))

            if "model" not in meta:
                meta["model"] = self._model

        return all_embeddings, meta

    async def _embed_batch_async(
        self, parts_to_embed: list[Part], batch_size: int
    ) -> tuple[list[list[float] | None], dict[str, Any]]:
        """
        Embed a list of parts in batches asynchronously.
        """

        resolved_config = EmbedContentConfig(**self._config) if self._config else None

        all_embeddings: list[list[float] | None] = []
        meta: dict[str, Any] = {}
        async for i in async_tqdm(
            range(0, len(parts_to_embed), batch_size), disable=not self._progress_bar, desc="Calculating embeddings"
        ):
            batch = parts_to_embed[i : i + batch_size]
            args: dict[str, Any] = {"model": self._model, "contents": [Content(parts=[p]) for p in batch]}
            if resolved_config:
                args["config"] = resolved_config

            response = await self._client.aio.models.embed_content(**args)

            embeddings: list[list[float] | None] = []
            if response.embeddings:
                for el in response.embeddings:
                    if el.values:
                        embeddings.append(el.values)
                    else:
                        logger.warning("An embedding in the API response has no values, setting it to None.")
                        embeddings.append(None)
                all_embeddings.extend(embeddings)
            else:
                logger.warning(
                    "No embeddings returned by the API for a batch of {count} document(s). "
                    "Their embeddings will be set to None.",
                    count=len(batch),
                )
                all_embeddings.extend([None] * len(batch))

            if "model" not in meta:
                meta["model"] = self._model

        return all_embeddings, meta

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, list[Document]] | dict[str, Any]:
        """
        Embeds a list of documents.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """

        parts_to_embed = self._extract_parts_to_embed(documents=documents)

        meta: dict[str, Any]
        embeddings, meta = self._embed_batch(parts_to_embed=parts_to_embed, batch_size=self._batch_size)

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(replace(doc, embedding=emb))

        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]] | dict[str, Any]:
        """
        Embeds a list of documents asynchronously.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """

        parts_to_embed = self._extract_parts_to_embed(documents=documents)

        embeddings, meta = await self._embed_batch_async(parts_to_embed=parts_to_embed, batch_size=self._batch_size)

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(replace(doc, embedding=emb))

        return {"documents": new_documents, "meta": meta}
