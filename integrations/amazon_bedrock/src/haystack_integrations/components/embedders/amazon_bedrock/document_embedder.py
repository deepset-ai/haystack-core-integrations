import json
from typing import Any, Dict, List, Literal, Optional

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_MODELS = [
    "amazon.titan-embed-text-v1",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
    "amazon.titan-embed-text-v2:0",
    "amazon.titan-embed-image-v1",
]


@component
class AmazonBedrockDocumentEmbedder:
    """
    A component for computing Document embeddings using Amazon Bedrock.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    import os
    from haystack.dataclasses import Document
    from haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockDocumentEmbedder

    os.environ["AWS_ACCESS_KEY_ID"] = "..."
    os.environ["AWS_SECRET_ACCESS_KEY_ID"] = "..."
    os.environ["AWS_DEFAULT_REGION"] = "..."

    embedder = AmazonBedrockDocumentEmbedder(
        model="cohere.embed-english-v3",
        input_type="search_document",
    )

    doc = Document(content="I love Paris in the winter.", meta={"name": "doc1"})

    result = embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.002, 0.032, 0.504, ...]
    ```
    """

    def __init__(
        self,
        model: Literal[
            "amazon.titan-embed-text-v1",
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3",
            "amazon.titan-embed-text-v2:0",
            "amazon.titan-embed-image-v1",
        ],
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
        boto3_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the AmazonBedrockDocumentEmbedder with the provided parameters. The parameters are passed to the
        Amazon Bedrock client.

        Note that the AWS credentials are not required if the AWS environment is configured correctly. These are loaded
        automatically from the environment or the AWS configuration file and do not need to be provided explicitly via
        the constructor. If the AWS environment is not configured users need to provide the AWS credentials via the
        constructor. Aside from model, three required parameters are `aws_access_key_id`, `aws_secret_access_key`,
         and `aws_region_name`.

        :param model: The embedding model to use. The model has to be specified in the format outlined in the Amazon
            Bedrock [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param batch_size: Number of Documents to encode at once.
            Only Cohere models support batch inference. This parameter is ignored for Amazon Titan models.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
            to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        :param boto3_config: The configuration for the boto3 client.
        :param kwargs: Additional parameters to pass for model inference. For example, `input_type` and `truncate` for
            Cohere models.
        :raises ValueError: If the model is not supported.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly.
        """

        if not model or model not in SUPPORTED_EMBEDDING_MODELS:
            msg = "Please provide a valid model from the list of supported models: " + ", ".join(
                SUPPORTED_EMBEDDING_MODELS
            )
            raise ValueError(msg)

        self.model = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.boto3_config = boto3_config
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

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [str(doc.meta[key]) for key in self.meta_fields_to_embed if doc.meta.get(key)]

            text_to_embed = self.embedding_separator.join([*meta_values_to_embed, doc.content or ""])

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_cohere(self, documents: List[Document]) -> List[Document]:
        """
        Internal method to embed Documents using Cohere models.
        Batch inference is supported.
        """

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        cohere_body = {
            "input_type": self.kwargs.get("input_type", "search_document"),  # mandatory parameter for Cohere models
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
                msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
                raise AmazonBedrockInferenceError(msg) from exception

            response_body = json.loads(response.get("body").read())
            all_embeddings.extend(response_body["embeddings"])

        for doc, emb in zip(documents, all_embeddings):
            doc.embedding = emb

        return documents

    def _embed_titan(self, documents: List[Document]) -> List[Document]:
        """
        Internal method to embed Documents using Amazon Titan models.
        NOTE: Batch inference is not supported, so embeddings are created one by one.
        """

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        all_embeddings = []
        for text in tqdm(texts_to_embed, disable=not self.progress_bar, desc="Creating embeddings"):
            body = {"inputText": text}
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

        for doc, emb in zip(documents, all_embeddings):
            doc.embedding = emb

        return documents

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Embed the provided `Document`s using the specified model.

        :param documents: The `Document`s to embed.
        :returns: A dictionary with the following keys:
            - `documents`: The `Document`s with the `embedding` field populated.
        :raises AmazonBedrockInferenceError: If the inference fails.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "AmazonBedrockDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the AmazonBedrockTextEmbedder."
            )
            raise TypeError(msg)

        if "cohere" in self.model:
            documents_with_embeddings = self._embed_cohere(documents=documents)
        elif "titan" in self.model:
            documents_with_embeddings = self._embed_titan(documents=documents)
        else:
            msg = f"Model {self.model} is not supported. Supported models are: {', '.join(SUPPORTED_EMBEDDING_MODELS)}."
            raise ValueError(msg)

        return {"documents": documents_with_embeddings}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            model=self.model,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            boto3_config=self.boto3_config,
            **self.kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)
