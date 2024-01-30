import json
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from haystack import component, default_from_dict, default_to_dict

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


@component
class AmazonBedrockGenerator:
    """
    Generator based on a Hugging Face model.
    This component provides an interface to generate text using a Hugging Face model that runs locally.

    Usage example:
    ```python
    from amazon_bedrock_haystack.generators.amazon_bedrock import AmazonBedrockGenerator

    generator = AmazonBedrockGenerator(
        model="anthropic.claude-v2",
        max_length=99,
        aws_access_key_id="...",
        aws_secret_access_key="...",
        aws_session_token="...",
        aws_profile_name="...",
        aws_region_name="..."
    )

    print(generator.run("Who is the best American actor?"))
    ```
    """

    SUPPORTED_MODEL_PATTERNS: ClassVar[Dict[str, Type[BedrockModelAdapter]]] = {
        r"amazon.titan-text.*": AmazonTitanAdapter,
        r"ai21.j2.*": AI21LabsJurassic2Adapter,
        r"cohere.command.*": CohereCommandAdapter,
        r"anthropic.claude.*": AnthropicClaudeAdapter,
        r"meta.llama2.*": MetaLlama2ChatAdapter,
    }

    def __init__(
        self,
        model: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        max_length: Optional[int] = 100,
        **kwargs,
    ):
        if not model:
            msg = "'model' cannot be None or empty string"
            raise ValueError(msg)
        self.model = model
        self.max_length = max_length

        try:
            session = self.get_aws_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_region_name=aws_region_name,
                aws_profile_name=aws_profile_name,
            )
            self.client = session.client("bedrock-runtime")
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        model_input_kwargs = kwargs
        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        model_max_length = kwargs.get("model_max_length", 4096)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # It is hard to determine which tokenizer to use for the SageMaker model
        # so we use GPT2 tokenizer which will likely provide good token count approximation
        self.prompt_handler = DefaultPromptHandler(
            model="gpt2",
            model_max_length=model_max_length,
            max_length=self.max_length or 100,
        )

        model_adapter_cls = self.get_model_adapter(model=model)
        if not model_adapter_cls:
            msg = f"AmazonBedrockGenerator doesn't support the model {model}."
            raise AmazonBedrockConfigurationError(msg)
        self.model_adapter = model_adapter_cls(model_kwargs=model_input_kwargs, max_length=self.max_length)

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        if isinstance(prompt, List):
            msg = (
                "AmazonBedrockGenerator only supports a string as a prompt, "
                "while currently, the prompt is of type List."
            )
            raise ValueError(msg)

        resize_info = self.prompt_handler(prompt)
        if resize_info["prompt_length"] != resize_info["new_prompt_length"]:
            logger.warning(
                "The prompt was truncated from %s tokens to %s tokens so that the prompt length and "
                "the answer length (%s tokens) fit within the model's max token limit (%s tokens). "
                "Shorten the prompt or it will be cut off.",
                resize_info["prompt_length"],
                max(0, resize_info["model_max_length"] - resize_info["max_length"]),  # type: ignore
                resize_info["max_length"],
                resize_info["model_max_length"],
            )
        return str(resize_info["resized_prompt"])

    @classmethod
    def supports(cls, model, **kwargs):
        model_supported = cls.get_model_adapter(model) is not None
        if not model_supported or not cls.aws_configured(**kwargs):
            return False

        try:
            session = cls.get_aws_session(**kwargs)
            bedrock = session.client("bedrock")
            foundation_models_response = bedrock.list_foundation_models(byOutputModality="TEXT")
            available_model_ids = [entry["modelId"] for entry in foundation_models_response.get("modelSummaries", [])]
            model_ids_supporting_streaming = [
                entry["modelId"]
                for entry in foundation_models_response.get("modelSummaries", [])
                if entry.get("responseStreamingSupported", False)
            ]
        except AWSConfigurationError as exception:
            raise AmazonBedrockConfigurationError(message=exception.message) from exception
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        model_available = model in available_model_ids
        if not model_available:
            msg = (
                f"The model {model} is not available in Amazon Bedrock. "
                f"Make sure the model you want to use is available in the configured AWS region and "
                f"you have access."
            )
            raise AmazonBedrockConfigurationError(msg)

        stream: bool = kwargs.get("stream", False)
        model_supports_streaming = model in model_ids_supporting_streaming
        if stream and not model_supports_streaming:
            msg = f"The model {model} doesn't support streaming. Remove the `stream` parameter."
            raise AmazonBedrockConfigurationError(msg)

        return model_supported

    def invoke(self, *args, **kwargs):
        kwargs = kwargs.copy()
        prompt: str = kwargs.pop("prompt", None)
        stream: bool = kwargs.get("stream", self.model_adapter.model_kwargs.get("stream", False))

        if not prompt or not isinstance(prompt, (str, list)):
            msg = (
                f"The model {self.model} requires a valid prompt, but currently, it has no prompt. "
                f"Make sure to provide a prompt in the format that the model expects."
            )
            raise ValueError(msg)

        body = self.model_adapter.prepare_body(prompt=prompt, **kwargs)
        try:
            if stream:
                response = self.client.invoke_model_with_response_stream(
                    body=json.dumps(body),
                    modelId=self.model,
                    accept="application/json",
                    contentType="application/json",
                )
                response_stream = response["body"]
                handler: TokenStreamingHandler = kwargs.get(
                    "stream_handler",
                    self.model_adapter.model_kwargs.get("stream_handler", DefaultTokenStreamingHandler()),
                )
                responses = self.model_adapter.get_stream_responses(stream=response_stream, stream_handler=handler)
            else:
                response = self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model,
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response.get("body").read().decode("utf-8"))
                responses = self.model_adapter.get_responses(response_body=response_body)
        except ClientError as exception:
            msg = (
                f"Could not connect to Amazon Bedrock model {self.model}. "
                f"Make sure your AWS environment is configured correctly, "
                f"the model is available in the configured AWS region, and you have access."
            )
            raise AmazonBedrockInferenceError(msg) from exception

        return responses

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        return {"replies": self.invoke(prompt=prompt, **(generation_kwargs or {}))}

    @classmethod
    def get_model_adapter(cls, model: str) -> Optional[Type[BedrockModelAdapter]]:
        for pattern, adapter in cls.SUPPORTED_MODEL_PATTERNS.items():
            if re.fullmatch(pattern, model):
                return adapter
        return None

    @classmethod
    def aws_configured(cls, **kwargs) -> bool:
        """
        Checks whether AWS configuration is provided.
        :param kwargs: The kwargs passed down to the generator.
        :return: True if AWS configuration is provided, False otherwise.
        """
        aws_config_provided = any(key in kwargs for key in AWS_CONFIGURATION_KEYS)
        return aws_config_provided

    @classmethod
    def get_aws_session(
        cls,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates an AWS Session with the given parameters.
        Checks if the provided AWS credentials are valid and can be used to connect to AWS.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param kwargs: The kwargs passed down to the service client. Supported kwargs depend on the model chosen.
            See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html.
        :raises AWSConfigurationError: If the provided AWS credentials are invalid.
        :return: The created AWS session.
        """
        try:
            return boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=aws_region_name,
                profile_name=aws_profile_name,
            )
        except BotoCoreError as e:
            provided_aws_config = {k: v for k, v in kwargs.items() if k in AWS_CONFIGURATION_KEYS}
            msg = f"Failed to initialize the session with provided AWS credentials {provided_aws_config}"
            raise AWSConfigurationError(msg) from e

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            max_length=self.max_length,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        return default_from_dict(cls, data)
