import logging
from typing import Any, Callable, Dict, List, Optional

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)


@component
class AmazonBedrockChatGenerator:
    """
    Completes chats using LLMs hosted on Amazon Bedrock.

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
        truncate: Optional[bool] = True,
        boto3_config: Optional[Dict[str, Any]] = None,
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
        :param generation_kwargs: Keyword arguments sent to the model. These
        parameters are specific to a model. You can find them in the model's documentation.
          For example, you can find the
        Anthropic Claude generation parameters in [Anthropic documentation](https://docs.anthropic.com/claude/reference/complete_post).
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
        :param boto3_config: The configuration for the boto3 client.

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
        self.truncate = truncate
        self.boto3_config = boto3_config

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
            config: Optional[Config] = None
            if self.boto3_config:
                config = Config(**self.boto3_config)
            self.client = session.client("bedrock-runtime", config=config)
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

        self.generation_kwargs = generation_kwargs or {}
        self.stop_words = stop_words or []
        self.streaming_callback = streaming_callback

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
            stop_words=self.stop_words,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
            truncate=self.truncate,
            boto3_config=self.boto3_config,
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
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        generation_kwargs = generation_kwargs or {}
        inference_config = self.generation_kwargs.copy()
        inference_config.update(generation_kwargs)

        # Prepare system prompts if any
        system_prompts: List[Dict[str, Any]] = []  # Type annotation added
        if messages and messages[0].is_from(ChatRole.SYSTEM):
            system_message = messages[0]
            system_prompts = [{"text": system_message.content}]
            messages = messages[1:]

        # Prepare messages
        messages_list = []
        for msg in messages:
            message_dict = {"role": msg.role.value, "content": [{"text": msg.content}]}
            messages_list.append(message_dict)

        try:
            if streaming_callback or self.streaming_callback:
                response = self.client.converse_stream(
                    modelId=self.model,
                    messages=messages_list,
                    system=system_prompts,  # Now properly typed
                    inferenceConfig=inference_config,
                )
                response_stream = response.get("stream")
                if not response_stream:
                    msg = "No stream found in the response."
                    raise AmazonBedrockInferenceError(msg)
                callback = streaming_callback or self.streaming_callback
                if callback is None:  # This should never happen due to the if condition above
                    msg = "No streaming callback provided"
                    raise ValueError(msg)
                replies = self.process_streaming_response(response_stream, callback)
            else:
                response = self.client.converse(
                    modelId=self.model,
                    messages=messages_list,
                    system=system_prompts,  # Now properly typed
                    inferenceConfig=inference_config,
                )
                replies = self.extract_replies_from_response(response)
        except ClientError as exception:
            msg = f"Could not generate inference for Amazon Bedrock model {self.model} due: {exception}"
            raise AmazonBedrockInferenceError(msg) from exception

        return {"replies": replies}

    def extract_replies_from_response(self, response_body: Dict[str, Any]) -> List[ChatMessage]:
        replies = []
        if "output" in response_body and "message" in response_body["output"]:
            message = response_body["output"]["message"]
            if message["role"] == "assistant":
                content_blocks = message["content"]
                text = ""
                for content_block in content_blocks:
                    if "text" in content_block:
                        text += content_block["text"]

                # Convert usage format from Bedrock to OpenAI format
                usage = response_body.get("usage", {})
                meta = {
                    "model": self.model,
                    "index": 0,
                    "finish_reason": response_body.get("stopReason"),
                    "usage": {
                        "prompt_tokens": usage.get("inputTokens", 0),
                        "completion_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                    },
                }

                replies.append(ChatMessage.from_assistant(content=text, meta=meta))
        return replies

    def process_streaming_response(
        self, response_stream, streaming_callback: Callable[[StreamingChunk], None]
    ) -> List[ChatMessage]:
        content = ""
        meta = {
            "model": self.model,
            "index": 0,
        }

        for event in response_stream:
            # if "messageStart" in event:
            #     role = event["messageStart"]["role"]
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                delta_text = delta.get("text", "")
                if delta_text:
                    content += delta_text
                    streaming_chunk = StreamingChunk(content=delta_text, meta=None)
                    streaming_callback(streaming_chunk)
            if "messageStop" in event:
                meta["finish_reason"] = event["messageStop"].get("stopReason")
            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    usage = metadata["usage"]
                    meta["usage"] = {
                        "prompt_tokens": usage.get("inputTokens", 0),
                        "completion_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                    }

        replies = [ChatMessage.from_assistant(content=content, meta=meta)]
        return replies
