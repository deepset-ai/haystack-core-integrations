import json
import logging
from typing import Any, Dict, List, Literal, Optional
from tqdm import tqdm

from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_MODELS = ["amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"]


@component
class AmazonBedrockDocumentEmbedder:

    def __init__(
        self,
        model: Literal["amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"],
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",        
        **kwargs,
    ):
        if not model or model not in SUPPORTED_EMBEDDING_MODELS:
            msg = "Please provide a valid model from the list of supported models: " + ", ".join(
                SUPPORTED_EMBEDDING_MODELS
            )
            raise ValueError(msg)

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
            self._client = session.client("bedrock-runtime")
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        self.model = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed
        self.embedding_separator = embedding_separator
        self.kwargs = kwargs

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if doc.meta.get(key)
            ]

            text_to_embed = self.embedding_separator.join([*meta_values_to_embed, doc.content or ""])

            texts_to_embed.append(text_to_embed)
        return texts_to_embed
    
    def _embed_cohere(self, documents: List[Document]) -> List[Document]:

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        cohere_body = {
                "input_type": self.kwargs.get("input_type", "search_query"),  # mandatory parameter for Cohere models
            }
        if truncate := self.kwargs.get("truncate"):
            cohere_body["truncate"] = truncate  # optional parameter for Cohere models

        all_embeddings = []
        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size), disable=not self.progress_bar, desc="Creating embeddings"
        ):
            batch = texts_to_embed[i : i + self.batch_size]
            body = {"texts": batch, **cohere_body}

            try:
                response = self._client.invoke_model(
                    body=json.dumps(body), modelId=self.model, accept="*/*", contentType="application/json"
                )
            except ClientError as exception:
                msg = (
                    f"Could not connect to Amazon Bedrock model {self.model}. "
                    f"Make sure your AWS environment is configured correctly, "
                    f"the model is available in the configured AWS region, and you have access."
                )
                raise AmazonBedrockInferenceError(msg) from exception

            response_body = json.loads(response.get("body").read())
            all_embeddings.extend(response_body["embeddings"])

        for doc, emb in zip(documents, all_embeddings):
            doc.embedding = emb

        return documents

    def _embed_titan(self, documents: List[Document]) -> List[Document]:
        # NOTE: Amazon Titan models do not support batch inference

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        all_embeddings = []
        for text in tqdm(texts_to_embed, disable=not self.progress_bar, desc="Creating embeddings"):
            body = {"inputText": text}
            try:
                response = self._client.invoke_model(
                    body=json.dumps(body), modelId=self.model, accept="*/*", contentType="application/json"
                )
            except ClientError as exception:
                msg = (
                    f"Could not connect to Amazon Bedrock model {self.model}. "
                    f"Make sure your AWS environment is configured correctly, "
                    f"the model is available in the configured AWS region, and you have access."
                )
                raise AmazonBedrockInferenceError(msg) from exception

            response_body = json.loads(response.get("body").read())
            embedding = response_body["embedding"]
            all_embeddings.append(embedding)

        for doc, emb in zip(documents, all_embeddings):
            doc.embedding = emb

        return documents            
        


    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "AmazonBedrockDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the AmazonBedrockTextEmbedder."
            )
        
        if "cohere" in self.model:
            documents_with_embeddings = self._embed_cohere(documents=documents)
        elif "titan" in self.model:
            documents_with_embeddings = self._embed_titan(documents=documents)

        return {"documents": documents_with_embeddings}