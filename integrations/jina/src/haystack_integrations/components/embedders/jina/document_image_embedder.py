# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _encode_image_to_base64,
    _extract_image_sources_info,
    _PDFPageInfo,
)
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


@component
class JinaDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Jina AI multimodal models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    The JinaDocumentImageEmbedder supports models from the jina-clip series and jina-embeddings-v4
    which can encode images into vector representations in the same embedding space as text.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.jina import JinaDocumentImageEmbedder

    # Make sure that the environment variable JINA_API_KEY is set

    embedder = JinaDocumentImageEmbedder(model="jina-clip-v2")

    documents = [
        Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
        Document(content="A photo of a dog", meta={"file_path": "dog.jpg"}),
    ]

    result = embedder.run(documents=documents)
    documents_with_embeddings = result["documents"]
    print(documents_with_embeddings[0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),  # noqa: B008
        model: str = "jina-clip-v2",
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 5,
    ):
        """
        Create a JinaDocumentImageEmbedder component.

        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable `JINA_API_KEY` (recommended).
        :param model: The name of the Jina multimodal model to use.
            Supported models include:
            - "jina-clip-v1"
            - "jina-clip-v2" (default)
            - "jina-embeddings-v4"
            Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param embedding_dimension: Number of desired dimensions for the embedding.
            Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
            Only supported by jina-embeddings-v4.
        :param image_size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time.
        :param batch_size: Number of images to send in each API request. Defaults to 5.
        """

        resolved_api_key = api_key.resolve_value()

        self.api_key = api_key
        self.model_name = model
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.embedding_dimension = embedding_dimension
        self.image_size = image_size
        self.batch_size = batch_size
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model_name,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            embedding_dimension=self.embedding_dimension,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JinaDocumentImageEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _extract_images_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Validates the input documents and extracts the images to embed in the format expected by the Jina API.

        :param documents:
            Documents to embed.

        :returns:
            List of images to embed in the format expected by the Jina API.

        :raises TypeError:
            If the input is not a list of `Documents`.
        :raises RuntimeError:
            If the conversion of some documents fails.
        """
        if not isinstance(documents, list) or not all(isinstance(d, Document) for d in documents):
            msg = (
                "JinaDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the JinaTextEmbedder."
            )
            raise TypeError(msg)

        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path
        )

        images_to_embed: List[Optional[str]] = [None] * len(documents)
        pdf_page_infos: List[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["mime_type"] == "application/pdf":
                # Store PDF documents for later processing
                page_number = image_source_info.get("page_number")
                if page_number is None:
                    msg = f"Page number is required for PDF document at index {doc_idx}"
                    raise ValueError(msg)
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

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Embed a list of image documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """

        if not documents:
            return {"documents": []}

        images_to_embed = self._extract_images_to_embed(documents)

        embeddings = []

        # Process images in batches
        for i in range(0, len(images_to_embed), self.batch_size):
            batch_images = images_to_embed[i : i + self.batch_size]

            # Prepare request parameters
            parameters: Dict[str, Any] = {}
            if self.embedding_dimension is not None:
                parameters["dimensions"] = self.embedding_dimension

            try:
                response = self._session.post(
                    JINA_API_URL,
                    json={
                        "input": batch_images,
                        "model": self.model_name,
                        **parameters,
                    },
                )
                resp = response.json()
            except Exception as e:
                msg = f"Error calling Jina API: {e}"
                raise RuntimeError(msg) from e

            if "data" not in resp:
                error_msg = resp.get("detail", "Unknown error occurred")
                msg = f"Jina API error: {error_msg}"
                raise RuntimeError(msg)

            batch_embeddings = [item["embedding"] for item in resp["data"]]
            embeddings.extend(batch_embeddings)

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
