# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Tuple

from botocore.config import Config
from botocore.exceptions import ClientError
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

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_MODELS = ["amazon.titan-embed-image-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"]


@component
class AmazonBedrockDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Amazon Bedrock models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    ### Usage example
    ```python
    from haystack import Document
    rom haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockDocumentImageEmbedder

    os.environ["AWS_ACCESS_KEY_ID"] = "..."
    os.environ["AWS_SECRET_ACCESS_KEY_ID"] = "..."
    os.environ["AWS_DEFAULT_REGION"] = "..."

    embedder = AmazonBedrockDocumentImageEmbedder(model="amazon.titan-embed-image-v1")

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
    #           embedding=vector of size 512),
    #  ...]
    ```
    """

    def __init__(
        self,
        *,
        model: Literal["amazon.titan-embed-image-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"],
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        progress_bar: bool = True,
        boto3_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Creates a AmazonBedrockDocumentImageEmbedder component.

        :param model:
            The Bedrock model to use for calculating embeddings. Pass a valid model ID.
            Supported models:
            - "amazon.titan-embed-image-v1"
            - "cohere.embed-english-v3"
            - "cohere.embed-multilingual-v3"
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param image_size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param progress_bar:
            If `True`, shows a progress bar when embedding documents.
        :param boto3_config: The configuration for the boto3 client.
        :param kwargs: Additional parameters to pass for model inference.
            For example, `embeddingConfig` for Amazon Titan models and
            `embedding_types` for Cohere models.
        :raises ValueError: If the model is not supported.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly.
        """
        if not model or model not in SUPPORTED_EMBEDDING_MODELS:
            msg = "Please provide a valid model from the list of supported models: " + ", ".join(
                SUPPORTED_EMBEDDING_MODELS
            )
            raise ValueError(msg)

        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.model = model
        self.boto3_config = boto3_config

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.image_size = image_size
        self.progress_bar = progress_bar
        self.kwargs = kwargs
        self.embedding_types = None

        if emmbedding_types := self.kwargs.get("embedding_types"):
            if len(emmbedding_types) > 1:
                msg = (
                    "You have provided multiple embedding_types for Cohere model. "
                    "AmazonBedrockDocumentImageEmbedder only supports one embedding_type at a time."
                )
                raise ValueError(msg)
            self.embedding_types = emmbedding_types

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            config = Config(
                user_agent_extra="x-client-framework:haystack", **(self.boto3_config if self.boto3_config else {})
            )
            self._client = session.client("bedrock-runtime", config=config)
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

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
            model=self.model,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            progress_bar=self.progress_bar,
            boto3_config=self.boto3_config,
            image_size=self.image_size,
            **self.kwargs,
        )
        return serialization_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmazonBedrockDocumentImageEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(
            init_params,
            keys=[
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region_name",
                "aws_profile_name",
            ],
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed a list of images.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "AmazonBedrockDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the AmazonBedrockTextEmbedder."
            )
            raise TypeError(msg)
        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path
        )

        images_to_embed: list = [None] * len(documents)
        pdf_page_infos: list[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["mime_type"] == "application/pdf":
                # Store PDF documents for later processing
                page_number = image_source_info.get("page_number")

                pdf_page_info: _PDFPageInfo = {
                    "doc_idx": doc_idx,
                    "path": image_source_info["path"],
                    # page_number is added but mypy doesn't know that
                    "page_number": page_number,  # type: ignore[typeddict-item]
                }
                pdf_page_infos.append(pdf_page_info)
            else:
                # Process images directly
                image_byte_stream = ByteStream.from_file_path(
                    filepath=image_source_info["path"], mime_type=image_source_info["mime_type"]
                )
                mime_type, base64_image = _encode_image_to_base64(bytestream=image_byte_stream, size=self.image_size)
                if "cohere" in self.model:
                    images_to_embed[doc_idx] = f"data:{mime_type};base64,{base64_image}"
                else:
                    images_to_embed[doc_idx] = base64_image

        pdf_images_by_doc_idx = _batch_convert_pdf_pages_to_images(
            pdf_page_infos=pdf_page_infos, return_base64=True, size=self.image_size
        )

        # the pdf_images_by_doc_idx has base64 images but mypy cant detect that
        for doc_idx, base64_image in pdf_images_by_doc_idx.items():  # type: ignore[assignment]
            pdf_image_uri = f"data:application/pdf;base64,{base64_image}" if "cohere" in self.model else base64_image
            images_to_embed[doc_idx] = pdf_image_uri

        none_images_doc_ids = [documents[doc_idx].id for doc_idx, image in enumerate(images_to_embed) if image is None]
        if none_images_doc_ids:
            msg = f"Conversion failed for some documents. Document IDs: {none_images_doc_ids}."
            raise RuntimeError(msg)

        if "cohere" in self.model:
            embeddings = self._embed_cohere(image_uris=images_to_embed)
        elif "titan" in self.model:
            embeddings = self._embed_titan(images=images_to_embed)
        else:
            msg = f"Model {self.model} is not supported. Supported models are: {', '.join(SUPPORTED_EMBEDDING_MODELS)}."
            raise ValueError(msg)

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

    def _embed_titan(self, images: List[str]) -> List[List[float]]:
        """
        Internal method to embed base64 images using Amazon Titan models.

        :param images: List of base64 images.
        :return: List of embeddings.
        """

        titan_body = {}
        if embedding_config := self.kwargs.get("embeddingConfig"):
            titan_body["embeddingConfig"] = embedding_config  # optional parameter for Amazon Titan models

        all_embeddings = []

        for image in tqdm(images, disable=not self.progress_bar, desc="Creating embeddings"):
            body = {"inputImage": image, **titan_body}
            try:
                response = self._client.invoke_model(
                    body=json.dumps(body), modelId=self.model, accept="*/*", contentType="application/json"
                )
            except ClientError as exception:
                msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
                raise AmazonBedrockInferenceError(msg) from exception

            response_body = json.loads(response.get("body").read())
            embedding = response_body["embedding"]
            all_embeddings.append(embedding)

        return all_embeddings

    def _embed_cohere(self, image_uris: List[str]) -> List[List[float]]:
        """
        Internal method to embed base64 images using Cohere models.

        :param image_uris: List of image uris containing the base64 image and the mime type.
        :return: List of embeddings.
        """

        cohere_body = {"input_type": "image"}
        if self.embedding_types:
            cohere_body["embedding_types"] = self.embedding_types

        all_embeddings = []

        for image in tqdm(image_uris, disable=not self.progress_bar, desc="Creating embeddings"):
            body = {"images": [image], **cohere_body}
            try:
                response = self._client.invoke_model(
                    body=json.dumps(body), modelId=self.model, accept="*/*", contentType="application/json"
                )
            except ClientError as exception:
                msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
                raise AmazonBedrockInferenceError(msg) from exception

            response_body = json.loads(response.get("body").read())
            embeddings = response_body["embeddings"]

            # if embedding_types is specified, cohere returns a dict with the embedding types as keys
            if isinstance(embeddings, dict):
                for embedding in embeddings.values():
                    all_embeddings.append(embedding[0])
            else:
                # if embedding_types is not specified, cohere returns
                # a nested list of float embeddings
                all_embeddings.append(embeddings[0])

        return all_embeddings
