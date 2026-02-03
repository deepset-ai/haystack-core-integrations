import json
from typing import Any

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils.auth import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)


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
    os.environ["AWS_DEFAULT_REGION"] = "..."

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
        model: str,
        aws_access_key_id: Secret | None = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Secret | None = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Secret | None = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Secret | None = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Secret | None = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        boto3_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the AmazonBedrockTextEmbedder with the provided parameters. The parameters are passed to the
        Amazon Bedrock client.

        Note that the AWS credentials are not required if the AWS environment is configured correctly. These are loaded
        automatically from the environment or the AWS configuration file and do not need to be provided explicitly via
        the constructor. If the AWS environment is not configured users need to provide the AWS credentials via the
        constructor. Aside from model, three required parameters are `aws_access_key_id`, `aws_secret_access_key`,
         and `aws_region_name`.

        :param model: The embedding model to use.
            Amazon Titan and Cohere embedding models are supported, for example:
            "amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2:0", "amazon.titan-embed-image-v1",
            "cohere.embed-english-v3", "cohere.embed-multilingual-v3", "cohere.embed-v4:0".
            To find all supported models, refer to the Amazon Bedrock
            [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) and
            filter for "embedding", then select models from the Amazon Titan and Cohere series.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param boto3_config: The configuration for the boto3 client.
        :param kwargs: Additional parameters to pass for model inference. For example, `input_type` and `truncate` for
            Cohere models.
        :raises ValueError: If the model is not supported.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly.
        """
        if "titan" not in model and "cohere" not in model:
            msg = f"Model {model} is not supported. Only Amazon Titan and Cohere embedding models are supported."
            raise ValueError(msg)

        self.model = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.boto3_config = boto3_config
        self.kwargs = kwargs

        def resolve_secret(secret: Secret | None) -> str | None:
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

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
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
            msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        response_body = json.loads(response.get("body").read())

        if "cohere" in self.model:
            cohere_embeddings = response_body["embeddings"]
            # depending on the model, Cohere returns a dict with the embedding types as keys or a list of lists
            embeddings_list = (
                next(iter(cohere_embeddings.values())) if isinstance(cohere_embeddings, dict) else cohere_embeddings
            )
            embedding = embeddings_list[0]
        elif "titan" in self.model:
            embedding = response_body["embedding"]
        else:
            msg = f"Model {self.model} is not supported. Only Amazon Titan and Cohere embedding models are supported."
            raise ValueError(msg)

        return {"embedding": embedding}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            aws_region_name=self.aws_region_name,
            aws_profile_name=self.aws_profile_name,
            model=self.model,
            boto3_config=self.boto3_config,
            **self.kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmazonBedrockTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
