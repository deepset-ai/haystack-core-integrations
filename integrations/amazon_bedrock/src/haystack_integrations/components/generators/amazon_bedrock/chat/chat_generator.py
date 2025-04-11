from typing import Any, Dict, List, Optional, Tuple

from botocore.config import Config
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingCallbackT, select_streaming_callback
from haystack.tools import Tool, _check_duplicate_tool_names
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

# Compatibility with Haystack 2.12.0 and 2.13.0 - remove after 2.13.0 is released
try:
    from haystack.tools import deserialize_tools_or_toolset_inplace
except ImportError:
    from haystack.tools import deserialize_tools_inplace as deserialize_tools_or_toolset_inplace

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session
from haystack_integrations.components.generators.amazon_bedrock.chat.utils import (
    _format_messages,
    _format_tools,
    _parse_completion_response,
    _parse_streaming_response,
    _parse_streaming_response_async,
)

logger = logging.getLogger(__name__)


@component
class AmazonBedrockChatGenerator:
    """
    Completes chats using LLMs hosted on Amazon Bedrock available via the Bedrock Converse API.

    For example, to use the Anthropic Claude 3 Sonnet model, initialize this component with the
    'anthropic.claude-3-5-sonnet-20240620-v1:0' model name.

    ### Usage example

    ```python
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.components.generators.utils import print_streaming_chunk

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant, answer in German only"),
                ChatMessage.from_user("What's Natural Language Processing?")]


    client = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                                        streaming_callback=print_streaming_chunk)
    client.run(messages, generation_kwargs={"max_tokens": 512})
    ```

    ### Tool usage example
    # AmazonBedrockChatGenerator supports Haystack's unified tool architecture, allowing tools to be used
    # across different chat generators. The same tool definitions and usage patterns work consistently
    # whether using Amazon Bedrock, OpenAI, Ollama, or any other supported LLM providers.

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

    def weather(city: str):
        return f'The weather in {city} is sunny and 32°C'

    # Define tool parameters
    tool_parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }

    # Create weather tool
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=weather
    )

    # Initialize generator with tool
    client = AmazonBedrockChatGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        tools=[weather_tool]
    )

    # Run initial query
    messages = [ChatMessage.from_user("What's the weather like in Paris?")]
    results = client.run(messages=messages)

    # Get tool call from response
    tool_message = next(msg for msg in results["replies"] if msg.tool_call)
    tool_call = tool_message.tool_call

    # Execute tool and send result back
    weather_result = weather(**tool_call.arguments)
    new_messages = [
        messages[0],
        tool_message,
        ChatMessage.from_tool(tool_result=weather_result, origin=tool_call)
    ]

    # Get final response
    final_result = client.run(new_messages)
    print(final_result["replies"][0].text)

    > Based on the information I've received, I can tell you that the weather in Paris is
    > currently sunny with a temperature of 32°C (which is about 90°F).

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
        streaming_callback: Optional[StreamingCallbackT] = None,
        boto3_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Initializes the `AmazonBedrockChatGenerator` with the provided parameters. The parameters are passed to the
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
        :param generation_kwargs: Keyword arguments sent to the model. These parameters are specific to a model.
            You can find the model specific arguments in the AWS Bedrock API
            [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html).
        :param stop_words: A list of stop words that stop the model from generating more text
            when encountered. You can provide them using this parameter or using the model's `generation_kwargs`
            under a model's specific key for stop words.
            For example, you can provide stop words for Anthropic Claude in the `stop_sequences` key.
        :param streaming_callback: A callback function called when a new token is received from the stream.
            By default, the model is not set up for streaming. To enable streaming, set this parameter to a callback
            function that handles the streaming chunks. The callback function receives a
            [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk) object and switches
            the streaming mode on.
        :param boto3_config: The configuration for the boto3 client.
        :param tools: A list of Tool objects that the model can use. Each tool should have a unique name.

        :raises ValueError: If the model name is empty or None.
        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly or the model is
            not supported.
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
        self.stop_words = stop_words or []
        self.streaming_callback = streaming_callback
        self.boto3_config = boto3_config
        _check_duplicate_tool_names(tools)
        self.tools = tools

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        aws_access_key_id = resolve_secret(aws_access_key_id)
        aws_secret_access_key = resolve_secret(aws_secret_access_key)
        aws_session_token = resolve_secret(aws_session_token)
        aws_region_name = resolve_secret(aws_region_name)
        aws_profile_name = resolve_secret(aws_profile_name)

        config: Optional[Config] = None
        if self.boto3_config:
            config = Config(**self.boto3_config)

        try:
            # sync session
            session = get_aws_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_region_name=aws_region_name,
                aws_profile_name=aws_profile_name,
            )
            self.client = session.client("bedrock-runtime", config=config)

        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        self.generation_kwargs = generation_kwargs or {}
        self.stop_words = stop_words or []
        self.async_session = None

    def _get_async_session(self):
        if self.async_session:
            return self.async_session

        try:
            self.async_session = get_aws_session(
                aws_access_key_id=self.aws_access_key_id.resolve_value() if self.aws_access_key_id else None,
                aws_secret_access_key=(
                    self.aws_secret_access_key.resolve_value() if self.aws_secret_access_key else None
                ),
                aws_session_token=self.aws_session_token.resolve_value() if self.aws_session_token else None,
                aws_region_name=self.aws_region_name.resolve_value() if self.aws_region_name else None,
                aws_profile_name=self.aws_profile_name.resolve_value() if self.aws_profile_name else None,
                async_mode=True,
            )
            return self.async_session

        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        serialized_tools = [tool.to_dict() for tool in self.tools] if self.tools else None
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            model=self.model,
            stop_words=self.stop_words,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
            boto3_config=self.boto3_config,
            tools=serialized_tools,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary with serialized data.
        :returns:
            Instance of `AmazonBedrockChatGenerator`.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        return default_from_dict(cls, data)

    def _prepare_request_params(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        requires_async: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[StreamingCallbackT]]:
        """
        Prepares the request parameters for both sync and async run methods.

        :param messages: List of ChatMessage objects representing the conversation history.
        :param streaming_callback: Optional callback function for handling streaming responses.
        :param generation_kwargs: Optional dictionary of generation parameters.
        :param tools: Optional list of Tool objects that the model can use.
        :param requires_async: Boolean indicating whether the request is for async execution.
            This affects how the streaming callback is selected.
        :return: Tuple of (request parameters dict, callback function)
        """
        generation_kwargs = generation_kwargs or {}

        # Merge generation_kwargs with defaults
        merged_kwargs = self.generation_kwargs.copy()
        merged_kwargs.update(generation_kwargs)

        # Extract known inference parameters
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html
        inference_config = {
            key: merged_kwargs.pop(key, None)
            for key in ["maxTokens", "stopSequences", "temperature", "topP"]
            if key in merged_kwargs
        }

        # Handle tools - either toolConfig or Haystack Tool objects but not both
        tools = tools or self.tools
        _check_duplicate_tool_names(tools)
        tool_config = merged_kwargs.pop("toolConfig", None)
        if tools:
            # Format Haystack tools to Bedrock format
            tool_config = _format_tools(tools)

        # Any remaining kwargs go to additionalModelRequestFields
        additional_fields = merged_kwargs if merged_kwargs else None

        # Format messages to Bedrock format
        system_prompts, messages_list = _format_messages(messages)

        # Build API parameters
        params = {
            "modelId": self.model,
            "messages": messages_list,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }
        if tool_config:
            params["toolConfig"] = tool_config
        if additional_fields:
            params["additionalModelRequestFields"] = additional_fields

        callback = select_streaming_callback(
            init_callback=self.streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=requires_async,
        )

        return params, callback

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        params, callback = self._prepare_request_params(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            requires_async=False,
        )

        try:
            if callback:
                response = self.client.converse_stream(**params)
                response_stream: EventStream = response.get("stream")
                if not response_stream:
                    msg = "No stream found in the response."
                    raise AmazonBedrockInferenceError(msg)
                replies = _parse_streaming_response(response_stream, callback, self.model)
            else:
                response = self.client.converse(**params)
                replies = _parse_completion_response(response, self.model)
        except ClientError as exception:
            msg = f"Could not generate inference for Amazon Bedrock model {self.model} due: {exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Async version of the run method. Completes chats using LLMs hosted on Amazon Bedrock.

        :param messages: List of ChatMessage objects representing the conversation history.
        :param streaming_callback: Optional callback function for handling streaming responses.
        :param generation_kwargs: Optional dictionary of generation parameters.
        :param tools: Optional list of Tool objects that the model can use.
        :return: Dictionary containing the model's replies as a list of ChatMessage objects.
        """
        params, callback = self._prepare_request_params(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            requires_async=True,
        )

        try:
            session = self._get_async_session()
            # Note: https://aioboto3.readthedocs.io/en/latest/usage.html
            # we need to create a new client for each request
            async with session.client("bedrock-runtime", config=self.boto3_config) as async_client:
                if callback:
                    response = await async_client.converse_stream(**params)
                    response_stream: EventStream = response.get("stream")
                    if not response_stream:
                        msg = "No stream found in the response."
                        raise AmazonBedrockInferenceError(msg)
                    replies = await _parse_streaming_response_async(response_stream, callback, self.model)
                else:
                    response = await async_client.converse(**params)
                    replies = _parse_completion_response(response, self.model)

        except ClientError as exception:
            msg = f"Could not generate inference for Amazon Bedrock model {self.model} due: {exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies}
