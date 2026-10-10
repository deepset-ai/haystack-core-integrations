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
from haystack_integrations.components.embedders.amazon_bedrock.utils import (
    EmbeddingModelFamily,
    _resolve_model_family,
    _supports_titan_v2_params,
)

logger = logging.getLogger(__name__)


@component
class AmazonBedrockTextEmbedder:
    """
    A component for embedding strings using Amazon Bedrock.

    Usage example:
    ```python
    import os
    from haystack_integrations.components.embedders.amazon_bedrock import (
        AmazonBedrockTextEmbedder,
    )

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
        aws_region_name: Secret | str | None = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Secret | None = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        boto3_config: dict[str, Any] | None = None,
        *,
        model_family: EmbeddingModelFamily | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the AmazonBedrockTextEmbedder with the provided parameters.

        The parameters are passed to the Amazon Bedrock client.

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
            Inference profile IDs and ARNs are accepted too. If the ID doesn't name the model, set `model_family`.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param boto3_config: Dictionary of configuration options for the underlying Boto3 client.
            Can be used to tune [retry behavior](https://docs.aws.amazon.com/boto3/latest/guide/retries.html)
            and other low-level settings like timeouts and connection management.
        :param model_family: The model family that determines the request and response format, `"cohere"` or
            `"titan"`. Set it when `model` doesn't name the model, for example for an
            [application inference profile](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html)
            ARN. If not set, the family is detected from `model`.
            With `"titan"` and such an ARN, `dimensions` and `normalize` are sent as configured, so set them only if
            the profile uses Amazon Titan Text Embeddings V2.
        :param kwargs: Additional parameters to pass for model inference. For example, `input_type`, `truncate`, and
            `output_dimension` (Cohere Embed v4 only) for Cohere models, or `dimensions` and `normalize` for
            Amazon Titan Text Embeddings V2.
        :raises ValueError: If the model is not supported or `model_family` is invalid.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly.
        """
        self._resolved_model_family = _resolve_model_family(model, model_family)

        self.model = model
        self.model_family = model_family
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.boto3_config = boto3_config
        self.kwargs = kwargs

        self._client: Any = None

    def warm_up(self) -> None:
        """Create the Amazon Bedrock client."""
        if self._client is not None:
            return

        def resolve_secret(secret: Secret | str | None) -> str | None:
            return secret.resolve_value() if isinstance(secret, Secret) else secret

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(self.aws_access_key_id),
                aws_secret_access_key=resolve_secret(self.aws_secret_access_key),
                aws_session_token=resolve_secret(self.aws_session_token),
                aws_region_name=resolve_secret(self.aws_region_name),
                aws_profile_name=resolve_secret(self.aws_profile_name),
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

    def close(self) -> None:
        """Close the Amazon Bedrock client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        """
        Embeds the input text using the Amazon Bedrock model.

        :param text: The input text to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        :raises TypeError: If the input text is not a string.
        :raises AmazonBedrockInferenceError: If the model inference fails.
        """
        self.warm_up()
        if not isinstance(text, str):
            msg = (
                "AmazonBedrockTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the AmazonBedrockTextEmbedder."
            )
            raise TypeError(msg)

        if self._resolved_model_family == "cohere":
            body = {
                "texts": [text],
                "input_type": self.kwargs.get("input_type", "search_query"),  # mandatory parameter for Cohere models
            }
            if truncate := self.kwargs.get("truncate"):
                body["truncate"] = truncate  # optional parameter for Cohere models
            if (output_dimension := self.kwargs.get("output_dimension")) is not None:
                body["output_dimension"] = output_dimension  # optional parameter for Cohere Embed v4

        else:
            body = {
                "inputText": text,
            }
            if _supports_titan_v2_params(self.model):
                if (dimensions := self.kwargs.get("dimensions")) is not None:
                    body["dimensions"] = dimensions
                if (normalize := self.kwargs.get("normalize")) is not None:
                    body["normalize"] = normalize

        try:
            response = self._client.invoke_model(
                body=json.dumps(body), modelId=self.model, accept="*/*", contentType="application/json"
            )
        except ClientError as exception:
            msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        response_body = json.loads(response.get("body").read())

        if self._resolved_model_family == "cohere":
            cohere_embeddings = response_body["embeddings"]
            # depending on the model, Cohere returns a dict with the embedding types as keys or a list of lists
            embeddings_list = (
                next(iter(cohere_embeddings.values())) if isinstance(cohere_embeddings, dict) else cohere_embeddings
            )
            embedding = embeddings_list[0]
        else:
            embedding = response_body["embedding"]

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
            model_family=self.model_family,
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
