import json
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from .adapters import (
    AI21LabsJurassic2Adapter,
    AmazonTitanAdapter,
    AnthropicClaudeAdapter,
    BedrockModelAdapter,
    CohereCommandAdapter,
    MetaLlama2ChatAdapter,
)
from .errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
    AWSConfigurationError,
)
from .handlers import (
    DefaultPromptHandler,
    DefaultTokenStreamingHandler,
    TokenStreamingHandler,
)

logger = logging.getLogger(__name__)

AWS_CONFIGURATION_KEYS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_profile_name",
]

BEDROCK_EMBEDDING_MODELS = ["amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"]


@component
class AmazonBedrockGenerator:

    def __init__(
        self,
        model: str,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        prefix: str = "",
        suffix: str = "",
    ):
        if not model or model not in BEDROCK_EMBEDDING_MODELS:
            msg = f"AmazonBedrockGenerator supports the following models: {BEDROCK_EMBEDDING_MODELS}"
            raise AmazonBedrockConfigurationError(msg)
        self.model = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            session = self.get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self.client = session.client("bedrock-runtime")
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        model_input_kwargs = kwargs
        # We pop the model_max_length as it is not sent to the model but used to truncate the prompt if needed
        model_max_length = kwargs.get("model_max_length", 4096)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # It is hard to determine which tokenizer to use for the SageMaker model
        # so we use GPT2 tokenizer which will likely provide good token count approximation
        self.prompt_handler = DefaultPromptHandler(
            tokenizer="gpt2",
            model_max_length=model_max_length,
            max_length=self.max_length or 100,
        )

        model_adapter_cls = self.get_model_adapter(model=model)
        if not model_adapter_cls:
            msg = f"AmazonBedrockGenerator doesn't support the model {model}."
            raise AmazonBedrockConfigurationError(msg)
        self.model_adapter = model_adapter_cls(model_kwargs=model_input_kwargs, max_length=self.max_length)
