import json
import re
import warnings
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Type, Union, get_args

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

from .adapters import (
    AI21LabsJurassic2Adapter,
    AmazonTitanAdapter,
    AnthropicClaudeAdapter,
    BedrockModelAdapter,
    CohereCommandAdapter,
    CohereCommandRAdapter,
    MetaLlamaAdapter,
    MistralAdapter,
)

logger = logging.getLogger(__name__)


@component
class AmazonBedrockGenerator:
    """
    Generates text using models hosted on Amazon Bedrock.

    For example, to use the Anthropic Claude model, pass 'anthropic.claude-v2' in the `model` parameter.
    Provide AWS credentials either through the local AWS profile or directly through
    `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, and `aws_region_name` parameters.

    ### Usage example

    ```python
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

    generator = AmazonBedrockGenerator(
            model="anthropic.claude-v2",
            max_length=99
    )

    print(generator.run("Who is the best American actor?"))
    ```

    AmazonBedrockGenerator uses AWS for authentication. You can use the AWS CLI to authenticate through your IAM.
    For more information on setting up an IAM identity-based policy, see [Amazon Bedrock documentation]
    (https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html).
    If the AWS environment is configured correctly, the AWS credentials are not required as they're loaded
    automatically from the environment or the AWS configuration file.
    If the AWS environment is not configured, set `aws_access_key_id`, `aws_secret_access_key`,
    `aws_session_token`, and `aws_region_name` as environment variables or pass them as
     [Secret](https://docs.haystack.deepset.ai/v2.0/docs/secret-management) arguments. Make sure the region you set
    supports Amazon Bedrock.
    """

    SUPPORTED_MODEL_PATTERNS: ClassVar[Dict[str, Type[BedrockModelAdapter]]] = {
        r"([a-z]{2}\.)?amazon.titan-text.*": AmazonTitanAdapter,
        r"([a-z]{2}\.)?ai21.j2.*": AI21LabsJurassic2Adapter,
        r"([a-z]{2}\.)?cohere.command-[^r].*": CohereCommandAdapter,
        r"([a-z]{2}\.)?cohere.command-r.*": CohereCommandRAdapter,
        r"([a-z]{2}\.)?anthropic.claude.*": AnthropicClaudeAdapter,
        r"([a-z]{2}\.)?meta.llama.*": MetaLlamaAdapter,
        r"([a-z]{2}\.)?mistral.*": MistralAdapter,
    }

    SUPPORTED_MODEL_FAMILIES: ClassVar[Dict[str, Type[BedrockModelAdapter]]] = {
        "amazon.titan-text": AmazonTitanAdapter,
        "ai21.j2": AI21LabsJurassic2Adapter,
        "cohere.command": CohereCommandAdapter,
        "cohere.command-r": CohereCommandRAdapter,
        "anthropic.claude": AnthropicClaudeAdapter,
        "meta.llama": MetaLlamaAdapter,
        "mistral": MistralAdapter,
    }

    MODEL_FAMILIES = Literal[
        "amazon.titan-text",
        "ai21.j2",
        "cohere.command",
        "cohere.command-r",
        "anthropic.claude",
        "meta.llama",
        "mistral",
    ]

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
        max_length: Optional[int] = None,
        truncate: Optional[bool] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        boto3_config: Optional[Dict[str, Any]] = None,
        model_family: Optional[MODEL_FAMILIES] = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a new `AmazonBedrockGenerator` instance.

        :param model: The name of the model to use.
        :param aws_access_key_id: The AWS access key ID.
        :param aws_secret_access_key: The AWS secret access key.
        :param aws_session_token: The AWS session token.
        :param aws_region_name: The AWS region name. Make sure the region you set supports Amazon Bedrock.
        :param aws_profile_name: The AWS profile name.
        :param max_length: The maximum length of the generated text. This can also be set in the `kwargs` parameter
            by using the model specific parameter name.
        :param truncate: Deprecated. This parameter no longer has any effect.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param boto3_config: The configuration for the boto3 client.
        :param model_family: The model family to use. If not provided, the model adapter is selected based on the model
            name.
        :param kwargs: Additional keyword arguments to be passed to the model.
            You can find the model specific arguments in AWS Bedrock's
            [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html).
        These arguments are specific to the model. You can find them in the model's documentation.
        :raises ValueError: If the model name is empty or None.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly or the model is
            not supported.
        """
        if not model:
            msg = "'model' cannot be None or empty string"
            raise ValueError(msg)
        self.model = model

        if truncate is not None:
            msg = "The 'truncate' parameter no longer has any effect. No truncation will be performed."
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
        self.truncate = truncate

        self.max_length = max_length
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.streaming_callback = streaming_callback
        self.boto3_config = boto3_config
        self.kwargs = kwargs
        self.model_family = model_family

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
            self.client = session.client("bedrock-runtime", config=config)
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        model_input_kwargs = kwargs

        model_adapter_cls = self.get_model_adapter(model=model, model_family=model_family)
        self.model_adapter = model_adapter_cls(model_kwargs=model_input_kwargs, max_length=self.max_length)

    @component.output_types(replies=List[str], meta=Dict[str, Any])
    def run(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[List[str], Dict[str, Any]]]:
        """
        Generates a list of string response to the given prompt.

        :param prompt: The prompt to generate a response for.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Additional keyword arguments passed to the generator.
        :returns: A dictionary with the following keys:
            - `replies`: A list of generated responses.
            - `meta`: A dictionary containing response metadata.
        :raises ValueError: If the prompt is empty or None.
        :raises AmazonBedrockInferenceError: If the model cannot be invoked.
        """
        generation_kwargs = generation_kwargs or {}
        generation_kwargs = generation_kwargs.copy()
        streaming_callback = streaming_callback or self.streaming_callback
        generation_kwargs["stream"] = streaming_callback is not None

        body = self.model_adapter.prepare_body(prompt=prompt, **generation_kwargs)
        try:
            if streaming_callback:
                response = self.client.invoke_model_with_response_stream(
                    body=json.dumps(body),
                    modelId=self.model,
                    accept="application/json",
                    contentType="application/json",
                )
                response_stream = response["body"]
                replies = self.model_adapter.get_stream_responses(
                    stream=response_stream, streaming_callback=streaming_callback
                )
            else:
                response = self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.model,
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response.get("body").read().decode("utf-8"))
                replies = self.model_adapter.get_responses(response_body=response_body)

            metadata = response.get("ResponseMetadata", {})

        except ClientError as exception:
            msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies, "meta": metadata}

    @classmethod
    def get_model_adapter(cls, model: str, model_family: Optional[str] = None) -> Type[BedrockModelAdapter]:
        """
        Gets the model adapter for the given model.

        If `model_family` is provided, the adapter for the model family is returned.
        If `model_family` is not provided, the adapter is auto-detected based on the model name.

        :param model: The model name.
        :param model_family: The model family.
        :returns: The model adapter class, or None if no adapter is found.
        :raises AmazonBedrockConfigurationError: If the model family is not supported or the model cannot be
            auto-detected.
        """
        if model_family:
            if model_family not in cls.SUPPORTED_MODEL_FAMILIES:
                msg = f"Model family {model_family} is not supported. Must be one of {get_args(cls.MODEL_FAMILIES)}."
                raise AmazonBedrockConfigurationError(msg)
            return cls.SUPPORTED_MODEL_FAMILIES[model_family]

        for pattern, adapter in cls.SUPPORTED_MODEL_PATTERNS.items():
            if re.fullmatch(pattern, model):
                return adapter

        msg = (
            f"Could not auto-detect model family of {model}. "
            f"`model_family` parameter must be one of {get_args(cls.MODEL_FAMILIES)}. "
            f"We highly recommend using the `AmazonBedrockChatGenerator` instead. "
            f"It has additional support for Amazon's Nova Canvas, Nova Lite, "
            f"Nova Pro, DeepSeek's DeepSeek-R1, and more models. "
            f"See https://haystack.deepset.ai/integrations/amazon-bedrock"
        )
        raise AmazonBedrockConfigurationError(msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            model=self.model,
            max_length=self.max_length,
            streaming_callback=callback_name,
            boto3_config=self.boto3_config,
            model_family=self.model_family,
            **self.kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockGenerator":
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
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)
