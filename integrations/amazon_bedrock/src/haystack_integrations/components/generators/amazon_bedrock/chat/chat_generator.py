import json
import logging
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

from .adapters import AnthropicClaudeChatAdapter, BedrockModelChatAdapter, MetaLlama2ChatAdapter

logger = logging.getLogger(__name__)


@component
class AmazonBedrockChatGenerator:
    """
    `AmazonBedrockChatGenerator` enables text generation via Amazon Bedrock hosted chat LLMs.

    For example, to use the Anthropic Claude 3 Sonnet model, simply initialize the `AmazonBedrockChatGenerator` with the
    'anthropic.claude-3-sonnet-20240229-v1:0' model name.

    ```python
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.components.generators.utils import print_streaming_chunk

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant, answer in German only"),
                ChatMessage.from_user("What's Natural Language Processing?")]


    client = AmazonBedrockChatGenerator(model="anthropic.claude-3-sonnet-20240229-v1:0",
                                        streaming_callback=print_streaming_chunk)
    client.run(messages, generation_kwargs={"max_tokens": 512})

    ```

    If you prefer non-streaming mode, simply remove the `streaming_callback` parameter, capture the return value of the
    component's run method and the `AmazonBedrockChatGenerator` will return the response in a non-streaming mode.
    """

    SUPPORTED_MODEL_PATTERNS: ClassVar[Dict[str, Type[BedrockModelChatAdapter]]] = {
        r"anthropic.claude.*": AnthropicClaudeChatAdapter,
        r"meta.llama2.*": MetaLlama2ChatAdapter,
    }

    def __init__(
        self,
        model: str,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var(["AWS_ACCESS_KEY_ID"], strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            ["AWS_SECRET_ACCESS_KEY"], strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var(["AWS_SESSION_TOKEN"], strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var(["AWS_DEFAULT_REGION"], strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var(["AWS_PROFILE"], strict=False),  # noqa: B008
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initializes the `AmazonBedrockChatGenerator` with the provided parameters. The parameters are passed to the
        Amazon Bedrock client.

        Note that the AWS credentials are not required if the AWS environment is configured correctly. These are loaded
        automatically from the environment or the AWS configuration file and do not need to be provided explicitly via
        the constructor. If the AWS environment is not configured users need to provide the AWS credentials via the
        constructor. Aside from model, three required parameters are `aws_access_key_id`, `aws_secret_access_key`,
        and `aws_region_name`.

        :param model: The model to use for generation. The model must be available in Amazon Bedrock. The model has to
        be specified in the format outlined in the Amazon Bedrock [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html).
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param generation_kwargs: Additional generation keyword arguments passed to the model. The defined keyword
        parameters are specific to a specific model and can be found in the model's documentation. For example, the
        Anthropic Claude generation parameters can be found [here](https://docs.anthropic.com/claude/reference/complete_post).
        :param stop_words: A list of stop words that stop model generation when encountered. They can be provided via
        this parameter or via models generation_kwargs under a model's specific key for stop words. For example, the
        Anthropic Claude stop words are provided via the `stop_sequences` key.
        :param streaming_callback: A callback function that is called when a new chunk is received from the stream.
        By default, the model is not set up for streaming. To enable streaming simply set this parameter to a callback
        function that will handle the streaming chunks. The callback function will receive a StreamingChunk object and
        switch the streaming mode on.
        """
        if not model:
            msg = "'model' cannot be None or empty string"
            raise ValueError(msg)
        self.model = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name

        # get the model adapter for the given model
        model_adapter_cls = self.get_model_adapter(model=model)
        if not model_adapter_cls:
            msg = f"AmazonBedrockGenerator doesn't support the model {model}."
            raise AmazonBedrockConfigurationError(msg)
        self.model_adapter = model_adapter_cls(generation_kwargs or {})

        # create the AWS session and client
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

        self.stop_words = stop_words or []
        self.streaming_callback = streaming_callback

    def invoke(self, *args, **kwargs):
        """
        Invokes the Amazon Bedrock LLM with the given parameters. The parameters are passed to the Amazon Bedrock
        client.

        :param args: The positional arguments passed to the generator.
        :param kwargs: The keyword arguments passed to the generator.
        :returns: List of `ChatMessage` generated by LLM.
        """

        kwargs = kwargs.copy()
        messages: List[ChatMessage] = kwargs.pop("messages", [])
        # check if the prompt is a list of ChatMessage objects
        if not (
            isinstance(messages, list)
            and len(messages) > 0
            and all(isinstance(message, ChatMessage) for message in messages)
        ):
            msg = f"The model {self.model} requires a list of ChatMessage objects as a prompt."
            raise ValueError(msg)

        body = self.model_adapter.prepare_body(messages=messages, **{"stop_words": self.stop_words, **kwargs})
        try:
            if self.streaming_callback:
                response = self.client.invoke_model_with_response_stream(
                    body=json.dumps(body), modelId=self.model, accept="application/json", contentType="application/json"
                )
                response_stream = response["body"]
                responses = self.model_adapter.get_stream_responses(
                    stream=response_stream, stream_handler=self.streaming_callback
                )
            else:
                response = self.client.invoke_model(
                    body=json.dumps(body), modelId=self.model, accept="application/json", contentType="application/json"
                )
                response_body = json.loads(response.get("body").read().decode("utf-8"))
                responses = self.model_adapter.get_responses(response_body=response_body)
        except ClientError as exception:
            msg = f"Could not inference Amazon Bedrock model {self.model} due: {exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return responses

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Generates a list of `ChatMessage` response to the given messages using the Amazon Bedrock LLM.

        :param messages: The messages to generate a response to.
        :param generation_kwargs: Additional generation keyword arguments passed to the model.
        :returns: A dictionary with the following keys:
            - `replies`: The generated List of `ChatMessage` objects.
        """
        return {"replies": self.invoke(messages=messages, **(generation_kwargs or {}))}

    @classmethod
    def get_model_adapter(cls, model: str) -> Optional[Type[BedrockModelChatAdapter]]:
        """
        Returns the model adapter for the given model.

        :param model: The model to get the adapter for.
        :returns: The model adapter for the given model, or None if the model is not supported.
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
            stop_words=self.stop_words,
            generation_kwargs=self.model_adapter.generation_kwargs,
            streaming_callback=serialize_callable(self.streaming_callback),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)
