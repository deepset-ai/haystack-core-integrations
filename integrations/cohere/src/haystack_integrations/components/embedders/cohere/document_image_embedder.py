# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Optional, Tuple

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _encode_image_to_base64,
    _extract_image_sources_info,
    _PDFPageInfo,
)
from haystack.dataclasses import ByteStream
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from cohere import AsyncClientV2, ClientV2

from .embedding_types import EmbeddingTypes

# PDF is not officially supported, but we convert PDFs to JPEG images
SUPPORTED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "application/pdf"]


logger = logging.getLogger(__name__)


@component
class CohereDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Cohere models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    ### Usage example
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.cohere import CohereDocumentImageEmbedder

    embedder = CohereDocumentImageEmbedder(model="embed-v4.0")

    documents = [
        Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
        Document(content="A photo of a dog", meta={"file_path": "dog.jpg"}),
    ]

    result = embedder.run(documents=documents)
    documents_with_embeddings = result["documents"]
    print(documents_with_embeddings)

    # [Document(id=...,
    #           content='A photo of a cat',
    #           meta={'file_path': 'cat.jpg',
    #                 'embedding_source': {'type': 'image', 'file_path_meta_field': 'file_path'}},
    #           embedding=vector of size 1536),
    #  ...]
    ```
    """

    def __init__(
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "embed-v4.0",
        api_base_url: str = "https://api.cohere.com",
        timeout: float = 120.0,
        embedding_dimension: Optional[int] = None,
        embedding_type: EmbeddingTypes = EmbeddingTypes.FLOAT,
        progress_bar: bool = True,
    ) -> None:
        """
        Creates a CohereDocumentImageEmbedder component.

        :param file_path_meta_field:
            The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path:
            The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param image_size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param api_key:
            The Cohere API key.
        :param model:
            The Cohere model to use for calculating embeddings.
            Read [Cohere documentation](https://docs.cohere.com/docs/models#embed) for a list of all supported models.
        :param api_base_url:
            The Cohere API base URL.
        :param timeout:
            Request timeout in seconds.
        :param embedding_dimension:
            The dimension of the embeddings to return. Only valid for v4 and newer models.
            Read [Cohere API reference](https://docs.cohere.com/reference/embed) for a list possible values and
            supported models.
        :param embedding_type:
            The type of embeddings to return. Defaults to float embeddings.
            Specifying a type different from float is only supported for Embed v3.0 and newer models.
        :param progress_bar:
            Whether to show a progress bar or not. Can be helpful to disable in production deployments
            to keep the logs clean.
        """

        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.image_size = image_size
        self.model = model
        self.embedding_dimension = embedding_dimension
        self.embedding_type = embedding_type
        self.progress_bar = progress_bar

        self._api_key = api_key
        self._api_base_url = api_base_url
        self._timeout = timeout

        self._client = ClientV2(
            api_key=self._api_key.resolve_value(),
            base_url=self._api_base_url,
            timeout=self._timeout,
            client_name="haystack",
        )
        self._async_client = AsyncClientV2(
            api_key=self._api_key.resolve_value(),
            base_url=self._api_base_url,
            timeout=self._timeout,
            client_name="haystack",
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            image_size=self.image_size,
            model=self.model,
            progress_bar=self.progress_bar,
            api_key=self._api_key.to_dict(),
            api_base_url=self._api_base_url,
            timeout=self._timeout,
            embedding_dimension=self.embedding_dimension,
            embedding_type=self.embedding_type.value,
        )
        return serialization_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CohereDocumentImageEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        init_params["embedding_type"] = EmbeddingTypes.from_str(init_params["embedding_type"])

        return default_from_dict(cls, data)

    def _extract_images_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Validates the input documents and extracts the images to embed in the format expected by the Cohere API.

        :param documents:
            Documents to embed.

        :returns:
            List of images to embed in the format expected by the Cohere API.

        :raises TypeError:
            If the input is not a list of `Documents`.
        :raises ValueError:
            If the input contains unsupported image MIME types.
        :raises RuntimeError:
            If the conversion of some documents fails.
        """
        if not isinstance(documents, list) or not all(isinstance(d, Document) for d in documents):
            msg = (
                "CohereDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the CohereTextEmbedder."
            )
            raise TypeError(msg)

        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path
        )

        for img_info in images_source_info:
            if img_info["mime_type"] not in SUPPORTED_IMAGE_MIME_TYPES:
                msg = (
                    f"Unsupported image MIME type: {img_info['mime_type']}. "
                    f"Supported types are: {', '.join(SUPPORTED_IMAGE_MIME_TYPES)}"
                )
                raise ValueError(msg)

        images_to_embed: list[Optional[str]] = [None] * len(documents)
        pdf_page_infos: list[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["mime_type"] == "application/pdf":
                # Store PDF documents for later processing
                page_number = image_source_info.get("page_number")
                assert page_number is not None  # checked in _extract_image_sources_info but mypy doesn't know that
                pdf_page_info: _PDFPageInfo = {
                    "doc_idx": doc_idx,
                    "path": image_source_info["path"],
                    "page_number": page_number,
                }
                pdf_page_infos.append(pdf_page_info)
            else:
                # Process images directly
                image_byte_stream = ByteStream.from_file_path(
                    filepath=image_source_info["path"], mime_type=image_source_info["mime_type"]
                )
                mime_type, base64_image = _encode_image_to_base64(bytestream=image_byte_stream, size=self.image_size)
                images_to_embed[doc_idx] = f"data:{mime_type};base64,{base64_image}"

        base64_jpeg_images_by_doc_idx = _batch_convert_pdf_pages_to_images(
            pdf_page_infos=pdf_page_infos, return_base64=True, size=self.image_size
        )
        for doc_idx, base64_jpeg_image in base64_jpeg_images_by_doc_idx.items():
            images_to_embed[doc_idx] = f"data:image/jpeg;base64,{base64_jpeg_image}"

        none_images_doc_ids = [documents[doc_idx].id for doc_idx, image in enumerate(images_to_embed) if image is None]
        if none_images_doc_ids:
            msg = f"Conversion failed for some documents. Document IDs: {none_images_doc_ids}."
            raise RuntimeError(msg)

        # tested above that image is not None, but mypy doesn't know that
        return images_to_embed  # type: ignore[return-value]

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed a list of image documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """

        images_to_embed = self._extract_images_to_embed(documents)

        embeddings = []

        # The Cohere API only supports passing one image at a time
        for doc, image in tqdm(zip(documents, images_to_embed), desc="Embedding images", disable=not self.progress_bar):
            try:
                response = self._client.embed(
                    model=self.model,
                    images=[image],
                    input_type="image",
                    output_dimension=self.embedding_dimension,
                    embedding_types=[self.embedding_type.value],
                )
                embedding = getattr(response.embeddings, self.embedding_type.value)[0]
            except Exception as e:
                msg = f"Error embedding Document {doc.id}"
                raise RuntimeError(msg) from e

            embeddings.append(embedding)

        docs_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            # we store this information for later inspection
            new_meta = {
                **doc.meta,
                "embedding_source": {"type": "image", "file_path_meta_field": self.file_path_meta_field},
            }
            new_doc = replace(doc, meta=new_meta, embedding=emb)
            docs_with_embeddings.append(new_doc)

        return {"documents": docs_with_embeddings}

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously embed a list of image documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """

        images_to_embed = self._extract_images_to_embed(documents)

        embeddings = []

        # The Cohere API only supports passing one image at a time
        for doc, image in tqdm(zip(documents, images_to_embed), desc="Embedding images", disable=not self.progress_bar):
            try:
                response = await self._async_client.embed(
                    model=self.model,
                    images=[image],
                    input_type="image",
                    output_dimension=self.embedding_dimension,
                    embedding_types=[self.embedding_type.value],
                )
                embedding = getattr(response.embeddings, self.embedding_type.value)[0]
            except Exception as e:
                msg = f"Error embedding Document {doc.id}"
                raise RuntimeError(msg) from e

            embeddings.append(embedding)

        docs_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            # we store this information for later inspection
            new_meta = {
                **doc.meta,
                "embedding_source": {"type": "image", "file_path_meta_field": self.file_path_meta_field},
            }
            new_doc = replace(doc, meta=new_meta, embedding=emb)
            docs_with_embeddings.append(new_doc)

        return {"documents": docs_with_embeddings}
