import json
import logging
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import deserialize_callback_handler
from haystack.dataclasses import ChatMessage, StreamingChunk

from haystack_integrations.components.generators.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
    AWSConfigurationError,
)

from .adapters import AnthropicClaudeChatAdapter, BedrockModelChatAdapter, MetaLlama2ChatAdapter

logger = logging.getLogger(__name__)

AWS_CONFIGURATION_KEYS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_profile_name",
]


@component
class AmazonBedrockChatGenerator:
    """
    AmazonBedrockChatGenerator enables text generation via Amazon Bedrock chat hosted models. For example, to use
    the Anthropic Claude model, simply initialize the AmazonBedrockChatGenerator with the 'anthropic.claude-v2'
    model name.

    ```python
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.components.generators.utils import print_streaming_chunk

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]


    client = AmazonBedrockChatGenerator(model="anthropic.claude-v2", streaming_callback=print_streaming_chunk)
    client.run(messages, generation_kwargs={"max_tokens_to_sample": 512})

    ```

    If you prefer non-streaming mode, simply remove the `streaming_callback` parameter, capture the return value of the
    component's run method and the AmazonBedrockChatGenerator will return the response in a non-streaming mode.
    """

    SUPPORTED_MODEL_PATTERNS: ClassVar[Dict[str, Type[BedrockModelChatAdapter]]] = {
        r"anthropic.claude.*": AnthropicClaudeChatAdapter,
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
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initializes the AmazonBedrockChatGenerator with the provided parameters. The parameters are passed to the
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

        # get the model adapter for the given model
        model_adapter_cls = self.get_model_adapter(model=model)
        if not model_adapter_cls:
            msg = f"AmazonBedrockGenerator doesn't support the model {model}."
            raise AmazonBedrockConfigurationError(msg)
        self.model_adapter = model_adapter_cls(generation_kwargs or {})

        # create the AWS session and client
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

        self.stop_words = stop_words or []
        self.streaming_callback = streaming_callback

    def invoke(self, *args, **kwargs):
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

        body = self.model_adapter.prepare_body(messages=messages, stop_words=self.stop_words, **kwargs)
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

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        return {"replies": self.invoke(messages=messages, **(generation_kwargs or {}))}

    @classmethod
    def get_model_adapter(cls, model: str) -> Optional[Type[BedrockModelChatAdapter]]:
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
            stop_words=self.stop_words,
            generation_kwargs=self.model_adapter.generation_kwargs,
            streaming_callback=self.streaming_callback,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockChatGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)
