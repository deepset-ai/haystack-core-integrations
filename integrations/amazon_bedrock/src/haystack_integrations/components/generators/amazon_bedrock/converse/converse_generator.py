import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

from .capabilities import (
    MODEL_CAPABILITIES,
    ModelCapability,
)
from .utils import (
    ContentBlock,
    ConverseMessage,
    ImageBlock,
    StreamEvent,
    ToolConfig,
    get_stream_message,
)

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
    from haystack import Pipeline

    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockConverseGenerator
    from haystack_integrations.components.generators.amazon_bedrock.converse.utils import ConverseMessage, ToolConfig


    def get_current_weather(location: str, unit: str = "celsius") -> str:
        return f"The weather in {location} is 22 degrees {unit}."


    def get_current_time(timezone: str) -> str:
        return f"The current time in {timezone} is 14:30."


    generator = AmazonBedrockConverseGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        # streaming_callback=print,
    )

    # Create ToolConfig from functions
    tool_config = ToolConfig.from_functions([get_current_weather, get_current_time])

    # Convert ToolConfig to dict for use in the run method
    tool_config_dict = tool_config.to_dict()

    print("Tool Config:")
    print(tool_config_dict)

    pipeline = Pipeline()
    pipeline.add_component("generator", generator)

    result = pipeline.run(
        data={
            "generator": {
                "inference_config": {
                    "temperature": 0.1,
                    "maxTokens": 256,
                    "topP": 0.1,
                    "stopSequences": ["\\n"],
                },
                "messages": [
                    ConverseMessage.from_user(["What's the weather like in Paris and what time is it in New York?"]),
                ],
                "tool_config": tool_config_dict,
            },
        },
    )
    print(result)
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
        inference_config: Optional[Dict[str, Any]] = None,
        tool_config: Optional[ToolConfig] = None,
        streaming_callback: Optional[Callable[[StreamEvent], None]] = None,
        system_prompt: Optional[List[Dict[str, Any]]] = None,
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
        :param streaming_callback: A callback function called when a new token is received from the stream.
        By default, the model is not set up for streaming. To enable streaming, set this parameter to a callback
        function that handles the streaming chunks. The callback function receives a
          [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk) object and
        switches the streaming mode on.
        :param inference_config: A dictionary containing the inference configuration. The default value is None.
        :param tool_config: A dictionary containing the tool configuration. The default value is None.
        :param system_prompt: A list of dictionaries containing the system prompt. The default value is None.
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

        self.inference_config = inference_config
        self.tool_config = tool_config
        self.streaming_callback = streaming_callback
        self.system_prompt = system_prompt
        self.model_capabilities = self._get_model_capabilities(model)  # create the AWS session and client

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

    def _get_model_capabilities(self, model: str) -> Set[ModelCapability]:
        for pattern, capabilities in MODEL_CAPABILITIES.items():
            if re.match(pattern, model):
                return capabilities
        unsupported_model_error = ValueError(f"Unsupported model: {model}")
        raise unsupported_model_error

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
        streaming_callback: Optional[Callable[[StreamEvent], None]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        tool_config: Optional[ToolConfig] = None,
        system_prompt: Optional[List[Dict[str, Any]]] = None,
    ):
        streaming_callback = streaming_callback or self.streaming_callback
        system_prompt = system_prompt or self.system_prompt
        tool_config = tool_config or self.tool_config

        if not (
            isinstance(messages, list)
            and len(messages) > 0
            and all(isinstance(message, ConverseMessage) for message in messages)
        ):
            msg = f"The model {self.model} requires a list of ConverseMessage objects as input."
            raise ValueError(msg)

        # Check and filter messages based on model capabilities
        if ModelCapability.SYSTEM_PROMPTS not in self.model_capabilities and system_prompt:
            logger.warning(
                f"The model {self.model} does not support system prompts. The provided system_prompt will be ignored."
            )
            system_prompt = None

        if ModelCapability.VISION not in self.model_capabilities:
            messages = [
                ConverseMessage(
                    role=msg.role,
                    content=ContentBlock(
                        content=[item for item in msg.content.content if not isinstance(item, ImageBlock)]
                    ),
                )
                for msg in messages
            ]

        if ModelCapability.DOCUMENT_CHAT not in self.model_capabilities:
            logger.warning(
                f"The model {self.model} does not support document chat. This feature will not be available."
            )

        if ModelCapability.TOOL_USE not in self.model_capabilities and tool_config:
            logger.warning(f"The model {self.model} does not support tools. The provided tool_config will be ignored.")
            tool_config = None

        if ModelCapability.STREAMING_TOOL_USE not in self.model_capabilities and streaming_callback and tool_config:
            logger.warning(
                f"The model {self.model} does not support streaming tool use. "
                "Streaming will be disabled for tool calls."
            )

        request_kwargs = {
            "modelId": self.model,
            "inferenceConfig": inference_config or self.inference_config,
            "messages": [message.to_dict() for message in messages],
        }

        if tool_config:
            request_kwargs["toolConfig"] = tool_config.to_dict()
        if system_prompt:
            request_kwargs["system"] = {
                "text": system_prompt,
            }

        try:
            if streaming_callback and ModelCapability.CONVERSE_STREAM in self.model_capabilities:
                converse_response = self.client.converse_stream(**request_kwargs)
                response_stream = converse_response.get("stream")
                message, metadata = get_stream_message(
                    stream=response_stream,
                    streaming_callback=streaming_callback,
                )
            else:
                converse_response = self.client.converse(**request_kwargs)
                output = converse_response.get("output")
                if output is None:
                    response_output_missing_error = "Response does not contain 'output'"
                    raise KeyError(response_output_missing_error)
                message = ConverseMessage.from_dict(output.get("message"))
                if message is None:
                    response_output_missing_message_error = "Response 'output' does not contain 'message'"
                    raise KeyError(response_output_missing_message_error)
                metadata = converse_response

            return {
                "message": message,
                "usage": metadata.get("usage"),
                "metrics": metadata.get("metrics"),
                "guardrail_trace": (
                    metadata.get("trace") if ModelCapability.GUARDRAILS in self.model_capabilities else None
                ),
                "stop_reason": metadata.get("stopReason"),
            }

        except ClientError as exception:
            msg = f"Could not run inference on Amazon Bedrock model {self.model} due to: {exception}"
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
            system_prompt=self.system_prompt,
            inference_config=self.inference_config,
            tool_config=self.tool_config.to_dict() if self.tool_config else None,
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
        tool_config = data.get("init_parameters", {}).get("tool_config")
        if tool_config:
            data["init_parameters"]["tool_config"] = ToolConfig.from_dict(tool_config)
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