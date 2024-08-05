import json
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

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
    DefaultTokenStreamingHandler,
    TokenStreamingHandler,
)

logger = logging.getLogger(__name__)


@component
class AmazonBedrockGenerator:
    """
    `AmazonBedrockGenerator` enables text generation via Amazon Bedrock hosted LLMs.

    For example, to use the Anthropic Claude model, simply initialize the `AmazonBedrockGenerator` with the
    'anthropic.claude-v2' model name. Provide AWS credentials either via local AWS profile or directly via
    `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, and `aws_region_name` parameters.

    Usage example:
    ```python
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

    generator = AmazonBedrockGenerator(
            model="anthropic.claude-v2",
            max_length=99
    )

    print(generator.run("Who is the best American actor?"))
    ```
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
        **kwargs,
    ):
        """
        Create a new `AmazonBedrockGenerator` instance.

        :param model: The name of the model to use.
        :param aws_access_key_id: The AWS access key ID.
        :param aws_secret_access_key: The AWS secret access key.
        :param aws_session_token: The AWS session token.
        :param aws_region_name: The AWS region name.
        :param aws_profile_name: The AWS profile name.
        :param max_length: The maximum length of the generated text.
        :param truncate: Whether to truncate the prompt or not.
        :param kwargs: Additional keyword arguments to be passed to the model.
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

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """
        Ensures that the prompt and answer token lengths together are within the model_max_length specified during
        the initialization of the component.

        :param prompt: The prompt to be sent to the model.
        :returns: The resized prompt.
        """
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

    def invoke(self, *args, **kwargs):
        """
        Invokes the model with the given prompt.

        :param args: Additional positional arguments passed to the generator.
        :param kwargs: Additional keyword arguments passed to the generator.
        :returns: A list of generated responses (strings).
        """
        kwargs = kwargs.copy()
        prompt: str = kwargs.pop("prompt", None)
        stream: bool = kwargs.get("stream", self.model_adapter.model_kwargs.get("stream", False))

        if not prompt or not isinstance(prompt, (str, list)):
            msg = (
                f"The model {self.model} requires a valid prompt, but currently, it has no prompt. "
                f"Make sure to provide a prompt in the format that the model expects."
            )
            raise ValueError(msg)

        if self.truncate:
            prompt = self._ensure_token_limit(prompt)

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

    @component.output_types(replies=List[str])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Generates a list of string response to the given prompt.

        :param prompt: The prompt to generate a response for.
        :param generation_kwargs: Additional keyword arguments passed to the generator.
        :returns: A dictionary with the following keys:
            - `replies`: A list of generated responses.
        :raises ValueError: If the prompt is empty or None.
        :raises AmazonBedrockInferenceError: If the model cannot be invoked.
        """
        return {"replies": self.invoke(prompt=prompt, **(generation_kwargs or {}))}

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
        return default_from_dict(cls, data)
