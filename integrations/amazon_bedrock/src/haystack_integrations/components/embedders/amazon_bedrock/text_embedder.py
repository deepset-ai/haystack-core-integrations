import json
import logging
from typing import Any, Dict, List, Literal, Optional

from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

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
]


@component
class AmazonBedrockTextEmbedder:
    """
    A component for embedding strings using Amazon Bedrock.

    Usage example:
    ```python
    import os
    from haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockTextEmbedder

    os.environ["AWS_ACCESS_KEY_ID"] = "..."
    os.environ["AWS_SECRET_ACCESS_KEY_ID"] = "..."
    os.environ["AWS_REGION_NAME"] = "..."

    embedder = AmazonBedrockTextEmbedder(
        model="cohere.embed-english-v3",
        input_type="search_query",
    )

    print(text_embedder.run("I love Paris in the summer."))

    # {'embedding': [0.002, 0.032, 0.504, ...]}
    ```
    """

    def __init__(
        self,
        model: Literal[
            "amazon.titan-embed-text-v1",
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3",
            "amazon.titan-embed-text-v2:0",
        ],
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        **kwargs,
    ):
        """
        Initializes the AmazonBedrockTextEmbedder with the provided parameters. The parameters are passed to the
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
        self.kwargs = kwargs

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embeds the input text using the Amazon Bedrock model.

        :param text: The input text to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        :raises TypeError: If the input text is not a string.
        :raises AmazonBedrockInferenceError: If the model inference fails.
        """
        if not isinstance(text, str):
            msg = (
                "AmazonBedrockTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the AmazonBedrockTextEmbedder."
            )
            raise TypeError(msg)

        if "cohere" in self.model:
            body = {
                "texts": [text],
                "input_type": self.kwargs.get("input_type", "search_query"),  # mandatory parameter for Cohere models
            }
            if truncate := self.kwargs.get("truncate"):
                body["truncate"] = truncate  # optional parameter for Cohere models

        elif "titan" in self.model:
            body = {
                "inputText": text,
            }

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

        if "cohere" in self.model:
            embedding = response_body["embeddings"][0]
        elif "titan" in self.model:
            embedding = response_body["embedding"]

        return {"embedding": embedding}

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
            **self.kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockTextEmbedder":
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
