# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _extract_image_sources_info,
    _PDFPageInfo,
)
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_MODELS = [
    "amazon.titan-embed-image-v1",
]


@component
class AmazonBedrockDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Amazon Bedrock models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    ### Usage example
    ```python
    from haystack import Document
    rom haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockTextEmbedder

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
        model: Literal["amazon.titan-embed-image-v1",],
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
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
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param progress_bar:
            If `True`, shows a progress bar when embedding documents.
        :param boto3_config: The configuration for the boto3 client.
        :param kwargs: Additional parameters to pass for model inference.
            For example, `embeddingConfig` for Amazon Titan models and `input_type` and `truncate` for Cohere models.
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
        self.progress_bar = progress_bar
        self.kwargs = kwargs

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
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            aws_region_name=self.aws_region_name,
            aws_profile_name=self.aws_profile_name,
            progress_bar=self.progress_bar,
            boto3_config=self.boto3_config,
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
                "In case you want to embed a list of strings, please use the AmazonBedrockTextEmbedder."
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
                    "page_number": page_number,
                }
                pdf_page_infos.append(pdf_page_info)
            else:
                # Read image from file and encode it as base64 string.
                base64_image = self._get_base64_image_uri(image_source_info["path"], image_source_info["mime_type"])

                # Process images directly
                images_to_embed[doc_idx] = base64_image

        pdf_images_by_doc_idx = _batch_convert_pdf_pages_to_images(pdf_page_infos=pdf_page_infos, return_base64=True)
        for doc_idx, base64_image in pdf_images_by_doc_idx.items():  # type: ignore[assignment]
            images_to_embed[doc_idx] = base64_image

        none_images_doc_ids = [documents[doc_idx].id for doc_idx, image in enumerate(images_to_embed) if image is None]
        if none_images_doc_ids:
            msg = f"Conversion failed for some documents. Document IDs: {none_images_doc_ids}."
            raise RuntimeError(msg)

        embeddings = self._embed_titan(documents=images_to_embed)

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

    def _embed_titan(self, documents: List[str]) -> List[List[float]]:
        """
        Internal method to embed base64 images using Amazon Titan models.
        NOTE: Batch inference is not supported, so embeddings are created one by one.
        """

        titan_body = {}
        if embedding_config := self.kwargs.get("embeddingConfig"):
            titan_body["embeddingConfig"] = embedding_config  # optional parameter for Amazon Titan models

        all_embeddings = []
        
        for image in tqdm(documents, disable=not self.progress_bar, desc="Creating embeddings"):
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
    
    def embed_cohere(self, documents: List[str]) -> List[List[float]]:
        """
        Internal method to embed base64 images using Cohere models.
        NOTE: Batch inference is not supported, so embeddings are created one by one.
        """

        cohere_body = {"input_type": "image"}
        if emmbedding_types:=self.kwargs.get("embedding_types"):
            cohere_body["embedding_types"] = emmbedding_types

        all_embeddings = []
        body = {"images": documents, **cohere_body}
        
        try:
            response = self._client.invoke_model(
                body=json.dumps(body), modelId=self.model, accept="*/*", contentType="application/json")
        except ClientError as exception:
            msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        response_body = json.loads(response.get("body").read())
        embedding = response_body["embedding"]
        all_embeddings.append(embedding)

        return all_embeddings
    
    def _get_base64_image_uri(self, image_file_path: str, image_mime_type: str):
        with open(image_file_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
        if "cohere" in self.model:
            return f"data:{image_mime_type};base64,{base64_image}"
        else:
            return base64_image

