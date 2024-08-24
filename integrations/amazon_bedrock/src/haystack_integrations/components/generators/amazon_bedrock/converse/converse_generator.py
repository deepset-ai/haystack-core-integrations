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
from utils import ConverseMessage, ConverseStreamingChunk, get_stream_message

logger = logging.getLogger(__name__)


@component
class AmazonBedrockConverseGenerator:
    """
    Completes chats using LLMs hosted on Amazon Bedrock using the converse api.
    References: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html

    For example, to use the Anthropic Claude 3 Sonnet model, initialize this component with the
    'anthropic.claude-3-sonnet-20240229-v1:0' model name.

    ### Usage example

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

    AmazonBedrockChatGenerator uses AWS for authentication. You can use the AWS CLI to authenticate through your IAM.
    For more information on setting up an IAM identity-based policy, see [Amazon Bedrock documentation]
    (https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html).

    If the AWS environment is configured correctly, the AWS credentials are not required as they're loaded
    automatically from the environment or the AWS configuration file.
    If the AWS environment is not configured, set `aws_access_key_id`, `aws_secret_access_key`,
      and `aws_region_name` as environment variables or pass them as
     [Secret](https://docs.haystack.deepset.ai/v2.0/docs/secret-management) arguments. Make sure the region you set
    supports Amazon Bedrock.
    """

    # according to the list provided in the toolConfig arg: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax
    SUPPORTED_TOOL_MODEL_PATTERNS: ClassVar[List[str]] = [
        r"anthropic.claude-3.*",
        r"cohere.command-r.*",
        r"mistral.mistral-large.*",
    ]

    UNSUPPORTED_CHAT_MODEL_PATTERNS: ClassVar[List[str]] = [
        r"cohere.command-text.*",
        r"cohere.command-light.*",
        r"ai21.j2.*",
    ]

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
        # used for pipeline setup
        inference_config: Optional[Dict[str, Any]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[ConverseStreamingChunk], None]] = None,
    ):
        """
        Initializes the `AmazonBedrockConverseGenerator` with the provided parameters. The parameters are passed to the
        Amazon Bedrock client.

        Note that the AWS credentials are not required if the AWS environment is configured correctly. These are loaded
        automatically from the environment or the AWS configuration file and do not need to be provided explicitly via
        the constructor. If the AWS environment is not configured users need to provide the AWS credentials via the
        constructor. Aside from model, three required parameters are `aws_access_key_id`, `aws_secret_access_key`,
        and `aws_region_name`.

        :param model: The model to use for text generation. The model must be available in Amazon Bedrock and must
        be specified in the format outlined in the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html).
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name. Make sure the region you set supports Amazon Bedrock.
        :param aws_profile_name: AWS profile name.
        :param generation_kwargs: Keyword arguments sent to the model. These
        parameters are specific to a model. You can find them in the [converse documentation]().
        :param stop_words: A list of stop words that stop the model from generating more text
          when encountered. You can provide them using
        this parameter or using the model's `generation_kwargs` under a model's specific key for stop words.
          For example, you can provide
        stop words for Anthropic Claude in the `stop_sequences` key.
        :param streaming_callback: A callback function called when a new token is received from the stream.
        By default, the model is not set up for streaming. To enable streaming, set this parameter to a callback
        function that handles the streaming chunks. The callback function receives a
          [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk) object and
        switches the streaming mode on.
        :param truncate: Whether to truncate the prompt messages or not.
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

            self.inference_config = inference_config
            self.tool_config = tool_config
            self.streaming_callback = streaming_callback

        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

    @component.output_types(
        message=ConverseMessage,
        usage=Dict[str, Any],
        metrics=Dict[str, Any],
        guardrail_trace=Dict[str, Any],
        stop_reason=str,
    )
    def run(
        self,
        messages: List[ConverseMessage],
        streaming_callback: Optional[Callable[[ConverseStreamingChunk], None]] = None,
        inference_config: Dict[str, Any] = {},
        tool_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Generates a list of `ChatMessage` response to the given messages using the Amazon Bedrock LLM.

        :param messages: The messages to generate a response to.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Additional generation keyword arguments passed to the model.
        :returns: A dictionary with the following keys:
            - `replies`: The generated List of `ChatMessage` objects.
        """
        streaming_callback = streaming_callback or self.streaming_callback

        # warn and only keep last message if model does not support chat
        if re.match("|".join(self.UNSUPPORTED_CHAT_MODEL_PATTERNS), self.model) and len(messages) > 1:
            logging.warning(
                f"The model {self.model} does not support chat. Only the last message " "will be taken into account."
            )
            messages = messages[-1:]

        # check if the prompt is a list of ConverseMessage objects
        if not (
            isinstance(messages, list)
            and len(messages) > 0
            and all(isinstance(message, ConverseMessage) for message in messages)
        ):
            msg = f"The model {self.model} requires a list of ConverseMessage objects as a prompt."
            raise ValueError(msg)

        request_kwargs = {
            "modelId": self.model,
            "inferenceConfig": inference_config,
            "messages": [message.to_dict() for message in messages],
        }

        tool_config = tool_config or self.tool_config
        if tool_config is not None:
            request_kwargs["toolConfig"] = tool_config

        try:
            if streaming_callback:
                response = self.client.converse_stream(**request_kwargs)
                response_stream = response.get("stream")
                message, metadata = get_stream_message(stream=response_stream, streaming_callback=streaming_callback)
                return {
                    "message": message,
                    "usage": metadata.get("usage"),
                    "metrics": metadata.get("metrics"),
                    "guardrail_trace": metadata.get("trace"),
                    "stop_reason": metadata.get("stopReason"),
                }
            else:
                # toolConfig is optionnal but the converse api will fail if it is empty, so we can add it only if tool_config is not None
                response = self.client.converse(**request_kwargs)

                output = response.get("output")
                if output is None:
                    raise KeyError
                message = output.get("message")
                if message is None:
                    raise KeyError

                return {
                    "message": ConverseMessage.from_dict(message),
                    "usage": response.get("usage"),
                    "metrics": response.get("metrics"),
                    "guardrail_trace": response.get("trace"),
                    "stop_reason": response.get("stopReason"),
                }
        except ClientError as exception:
            msg = f"Could not inference Amazon Bedrock model {self.model} due: {exception}"
            raise AmazonBedrockInferenceError(msg) from exception

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
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockConverseGenerator":
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
            [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region_name",
                "aws_profile_name",
            ],
        )
        return default_from_dict(cls, data)
