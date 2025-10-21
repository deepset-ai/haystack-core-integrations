from typing import Any, Dict, List, Optional, Tuple

import aioboto3
from botocore.config import Config
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, ComponentInfo, StreamingCallbackT, select_streaming_callback
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

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
    _validate_guardrail_config,
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

    ### Multimodal example
    ```python
    from haystack.dataclasses import ChatMessage, ImageContent
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

    generator = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")

    image_content = ImageContent.from_file_path(file_path="apple.jpg")

    message = ChatMessage.from_user(content_parts=["Describe the image using 10 words at most.", image_content])

    response = generator.run(messages=[message])["replies"][0].text

    print(response)
    > The image shows a red apple.

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
        streaming_callback: Optional[StreamingCallbackT] = None,
        boto3_config: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        *,
        guardrail_config: Optional[Dict[str, str]] = None,
    ) -> None:
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
        :param streaming_callback: A callback function called when a new token is received from the stream.
            By default, the model is not set up for streaming. To enable streaming, set this parameter to a callback
            function that handles the streaming chunks. The callback function receives a
            [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk) object and switches
            the streaming mode on.
        :param boto3_config: The configuration for the boto3 client.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param guardrail_config: Optional configuration for a guardrail that has been created in Amazon Bedrock.
            This must be provided as a dictionary matching either
            [GuardrailConfiguration](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_GuardrailConfiguration.html).
            or, in streaming mode (when `streaming_callback` is set),
            [GuardrailStreamConfiguration](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailStreamConfiguration.html).
            If `trace` is set to `enabled`, the guardrail trace will be included under the `trace` key in the `meta`
            attribute of the resulting `ChatMessage`.
            Note: Enabling guardrails in streaming mode may introduce additional latency.
            To manage this, you can adjust the `streamProcessingMode` parameter.
            See the
            [Guardrails Streaming documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-streaming.html)
            for more information.


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
        self.streaming_callback = streaming_callback
        self.boto3_config = boto3_config

        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))
        self.tools = tools

        _validate_guardrail_config(guardrail_config=guardrail_config, streaming=streaming_callback is not None)
        self.guardrail_config = guardrail_config

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        config = Config(
            user_agent_extra="x-client-framework:haystack", **(self.boto3_config if self.boto3_config else {})
        )

        try:
            # sync session
            session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )

            self.client = session.client("bedrock-runtime", config=config)

        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        self.generation_kwargs = generation_kwargs or {}
        self.async_session: Optional[aioboto3.Session] = None

    def _get_async_session(self) -> aioboto3.Session:
        """
        Initializes and returns an asynchronous AWS session for accessing Amazon Bedrock.

        If the session is already created, it is reused. Otherwise, a new session is created using the provided AWS
        credentials and configuration.

        :returns:
            An async-compatible boto3 session configured for use with Amazon Bedrock.
        :raises AmazonBedrockConfigurationError:
            If unable to establish an async session due to misconfiguration.
        """
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
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            model=self.model,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
            boto3_config=self.boto3_config,
            tools=serialize_tools_or_toolset(self.tools),
            guardrail_config=self.guardrail_config,
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

        stop_words = init_params.pop("stop_words", None)
        msg = "stop_words parameter will be ignored. Use the `stopSequences` key in `generation_kwargs` instead."
        if stop_words:
            logger.warning(msg)

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
        tools: Optional[ToolsType] = None,
        requires_async: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[StreamingCallbackT]]:
        """
        Prepares and formats parameters required to call the Amazon Bedrock Converse API.

        This includes merging default and runtime generation parameters, formatting messages and tools, and
        selecting the appropriate streaming callback.

        :param messages: List of `ChatMessage` objects representing the conversation history.
        :param streaming_callback: Optional streaming callback provided at runtime.
        :param generation_kwargs: Optional dictionary of generation parameters. Some common parameters are:
            - `maxTokens`: Maximum number of tokens to generate.
            - `stopSequences`: List of stop sequences to stop generation.
            - `temperature`: Sampling temperature.
            - `topP`: Nucleus sampling parameter.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param requires_async: Boolean flag to indicate if an async-compatible streaming callback function is needed.

        :returns:
            A tuple of (API-ready parameter dictionary, streaming callback function).
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
        flattened_tools = flatten_tools_or_toolsets(tools)
        _check_duplicate_tool_names(flattened_tools)
        tool_config = merged_kwargs.pop("toolConfig", None)
        if flattened_tools:
            # Format Haystack tools to Bedrock format
            tool_config = _format_tools(flattened_tools)

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
        if self.guardrail_config:
            params["guardrailConfig"] = self.guardrail_config

        # overloads that exhaust finite Literals(bool) not treated as exhaustive
        # see https://github.com/python/mypy/issues/14764
        callback = select_streaming_callback(  # type: ignore[call-overload]
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
        tools: Optional[ToolsType] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Executes a synchronous inference call to the Amazon Bedrock model using the Converse API.

        Supports both standard and streaming responses depending on whether a streaming callback is provided.

        :param messages: A list of `ChatMessage` objects forming the chat history.
        :param streaming_callback: Optional callback for handling streaming outputs.
        :param generation_kwargs: Optional dictionary of generation parameters. Some common parameters are:
            - `maxTokens`: Maximum number of tokens to generate.
            - `stopSequences`: List of stop sequences to stop generation.
            - `temperature`: Sampling temperature.
            - `topP`: Nucleus sampling parameter.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.

        :returns:
            A dictionary containing the model-generated replies under the `"replies"` key.
        :raises AmazonBedrockInferenceError:
            If the Bedrock inference API call fails.
        """
        component_info = ComponentInfo.from_component(self)

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
                # the type of streaming callback is checked in _prepare_request_params, but mypy doesn't know
                replies = _parse_streaming_response(
                    response_stream=response_stream,
                    streaming_callback=callback,  # type: ignore[arg-type]
                    model=self.model,
                    component_info=component_info,
                )
            else:
                response = self.client.converse(**params)
                replies = _parse_completion_response(response, self.model)
        except ClientError as exception:
            msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Executes an asynchronous inference call to the Amazon Bedrock model using the Converse API.

        Designed for use cases where non-blocking or concurrent execution is desired.

        :param messages: A list of `ChatMessage` objects forming the chat history.
        :param streaming_callback: Optional async-compatible callback for handling streaming outputs.
        :param generation_kwargs: Optional dictionary of generation parameters. Some common parameters are:
            - `maxTokens`: Maximum number of tokens to generate.
            - `stopSequences`: List of stop sequences to stop generation.
            - `temperature`: Sampling temperature.
            - `topP`: Nucleus sampling parameter.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.

        :returns:
            A dictionary containing the model-generated replies under the `"replies"` key.
        :raises AmazonBedrockInferenceError:
            If the Bedrock inference API call fails.
        """
        component_info = ComponentInfo.from_component(self)

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
            config = Config(
                user_agent_extra="x-client-framework:haystack", **(self.boto3_config if self.boto3_config else {})
            )
            async with session.client("bedrock-runtime", config=config) as async_client:
                if callback:
                    response = await async_client.converse_stream(**params)
                    response_stream: EventStream = response.get("stream")
                    if not response_stream:
                        msg = "No stream found in the response."
                        raise AmazonBedrockInferenceError(msg)
                    # the type of streaming callback is checked in _prepare_request_params, but mypy doesn't know
                    replies = await _parse_streaming_response_async(
                        response_stream=response_stream,
                        streaming_callback=callback,  # type: ignore[arg-type]
                        model=self.model,
                        component_info=component_info,
                    )
                else:
                    response = await async_client.converse(**params)
                    replies = _parse_completion_response(response, self.model)

        except ClientError as exception:
            msg = f"Could not perform inference for Amazon Bedrock model {self.model} due to:\n{exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies}
