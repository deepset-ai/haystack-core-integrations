import json
import logging
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
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
from .handlers import (
    DefaultPromptHandler,
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
        r"amazon.titan-text.*": AmazonTitanAdapter,
        r"ai21.j2.*": AI21LabsJurassic2Adapter,
        r"cohere.command-[^r].*": CohereCommandAdapter,
        r"cohere.command-r.*": CohereCommandRAdapter,
        r"anthropic.claude.*": AnthropicClaudeAdapter,
        r"meta.llama.*": MetaLlamaAdapter,
        r"mistral.*": MistralAdapter,
    }

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
        max_length: Optional[int] = 100,
        truncate: Optional[bool] = True,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        **kwargs,
    ):
        """
        Create a new `AmazonBedrockGenerator` instance.

        :param model: The name of the model to use.
        :param aws_access_key_id: The AWS access key ID.
        :param aws_secret_access_key: The AWS secret access key.
        :param aws_session_token: The AWS session token.
        :param aws_region_name: The AWS region name. Make sure the region you set supports Amazon Bedrock.
        :param aws_profile_name: The AWS profile name.
        :param max_length: The maximum length of the generated text.
        :param truncate: Whether to truncate the prompt or not.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param kwargs: Additional keyword arguments to be passed to the model.
        These arguments are specific to the model. You can find them in the model's documentation.
        :raises ValueError: If the model name is empty or None.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly or the model is
            not supported.
        """
        if not model:
            msg = "'model' cannot be None or empty string"
            raise ValueError(msg)
        self.model = model
        self.max_length = max_length
        self.truncate = truncate
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.streaming_callback = streaming_callback
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
        # we use GPT2 tokenizer which will likely provide good token count approximation

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

    def _ensure_token_limit(self, prompt: str) -> str:
        """
        Ensures that the prompt and answer token lengths together are within the model_max_length specified during
        the initialization of the component.

        :param prompt: The prompt to be sent to the model.
        :returns: The resized prompt.
        """
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

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Generates a list of string response to the given prompt.

        :param prompt: The prompt to generate a response for.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Additional keyword arguments passed to the generator.
        :returns: A dictionary with the following keys:
            - `replies`: A list of generated responses.
        :raises ValueError: If the prompt is empty or None.
        :raises AmazonBedrockInferenceError: If the model cannot be invoked.
        """
        generation_kwargs = generation_kwargs or {}
        generation_kwargs = generation_kwargs.copy()
        streaming_callback = streaming_callback or self.streaming_callback
        generation_kwargs["stream"] = streaming_callback is not None

        if self.truncate:
            prompt = self._ensure_token_limit(prompt)

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
        except ClientError as exception:
            msg = (
                f"Could not connect to Amazon Bedrock model {self.model}. "
                f"Make sure your AWS environment is configured correctly, "
                f"the model is available in the configured AWS region, and you have access."
            )
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies}

    @classmethod
    def get_model_adapter(cls, model: str) -> Optional[Type[BedrockModelAdapter]]:
        """
        Gets the model adapter for the given model.

        :param model: The model name.
        :returns: The model adapter class, or None if no adapter is found.
        """
        for pattern, adapter in cls.SUPPORTED_MODEL_PATTERNS.items():
            if re.fullmatch(pattern, model):
                return adapter
        return None

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
            truncate=self.truncate,
            streaming_callback=callback_name,
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
