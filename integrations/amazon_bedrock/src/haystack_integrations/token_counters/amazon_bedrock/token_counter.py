# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.tools import ToolsType, flatten_tools_or_toolsets
from haystack.utils.auth import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

# Reuse the chat generator's Converse formatting so the count reflects exactly what
# `AmazonBedrockChatGenerator` would send to Bedrock for the same messages and tools.
from haystack_integrations.components.generators.amazon_bedrock.chat.utils import (
    _format_messages,
    _format_tools,
)


class AmazonBedrockTokenCounter:
    """
    Counts tokens with Amazon Bedrock's `CountTokens` API.

    Implements Haystack's `TokenCounter` protocol. Unlike local, tokenizer-based counters, this counter sends the
    input to Bedrock's `CountTokens` operation, so the returned count reflects the model's exact tokenization,
    including the formatting Bedrock applies to messages, system prompts, and tool schemas.

    The messages and tools are converted to the Bedrock `Converse` format (the same conversion the
    `AmazonBedrockChatGenerator` uses), so the count matches what an equivalent `Converse` request would consume.

    Because it delegates to a server-side API, `count()` measures a complete, valid conversation rather than an
    arbitrary set of messages: Bedrock validates the input the same way the `Converse` inference API does (it must
    begin with a user message, and tool results must pair with the tool calls that produced them). This is the right
    fit for sizing a whole request before sending it - its intended use - but it cannot size a stand-alone fragment
    such as a single tool-result message. For fragment-level counting (for example inside a compactor that measures
    individual messages), use a local counter such as `ApproximateTokenCounter`.

    ## Usage Example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.token_counters.amazon_bedrock import AmazonBedrockTokenCounter

    counter = AmazonBedrockTokenCounter(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
    messages = [ChatMessage.from_user("Hello, how are you?")]
    token_count = counter.count(messages)
    print(f"Token count: {token_count}")
    ```
    """

    def __init__(
        self,
        model: str,
        *,
        aws_access_key_id: Secret | None = Secret.from_env_var(["AWS_ACCESS_KEY_ID"], strict=False),  # noqa: B008
        aws_secret_access_key: Secret | None = Secret.from_env_var(  # noqa: B008
            ["AWS_SECRET_ACCESS_KEY"], strict=False
        ),
        aws_session_token: Secret | None = Secret.from_env_var(["AWS_SESSION_TOKEN"], strict=False),  # noqa: B008
        aws_region_name: Secret | str | None = Secret.from_env_var(["AWS_DEFAULT_REGION"], strict=False),  # noqa: B008
        aws_profile_name: Secret | None = Secret.from_env_var(["AWS_PROFILE"], strict=False),  # noqa: B008
        boto3_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the counter.

        :param model: The Bedrock model id (or ARN) whose tokenization should be used, for example
            `"anthropic.claude-3-5-sonnet-20240620-v1:0"`. Token counts are model-specific.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name. Make sure the region you set supports Amazon Bedrock.
        :param aws_profile_name: AWS profile name.
        :param boto3_config: Dictionary of configuration options for the underlying Boto3 client.
        :raises ValueError: If `model` is empty.
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
        self.boto3_config = boto3_config

        self.client: Any = None

    def warm_up(self) -> None:
        """
        Initialize the Amazon Bedrock client.

        :raises AmazonBedrockConfigurationError: If the AWS environment is not configured correctly.
        """
        if self.client is not None:
            return

        def resolve_secret(secret: Secret | str | None) -> str | None:
            return secret.resolve_value() if isinstance(secret, Secret) else secret

        config = Config(
            user_agent_extra="x-client-framework:haystack",
            **(self.boto3_config if self.boto3_config else {}),
        )

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(self.aws_access_key_id),
                aws_secret_access_key=resolve_secret(self.aws_secret_access_key),
                aws_session_token=resolve_secret(self.aws_session_token),
                aws_region_name=resolve_secret(self.aws_region_name),
                aws_profile_name=resolve_secret(self.aws_profile_name),
            )
            self.client = session.client("bedrock-runtime", config=config)
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Return the number of input tokens Bedrock will use for the given messages and tools.

        `messages` must form a complete, valid conversation: Bedrock validates it the same way the `Converse`
        inference API does (it must begin with a user message, and tool results must pair with their tool calls).
        To size an arbitrary fragment such as a single message, use a local counter like `ApproximateTokenCounter`.

        :param messages: The messages to measure.
        :param tools: Tools whose schemas are sent alongside the messages, and so consume tokens too. Pass them to
            have them counted; leave as None to measure the messages alone.
        :returns: The token count, or `0` when there is nothing to measure.
        :raises AmazonBedrockInferenceError: If the Bedrock `CountTokens` request fails.
        """
        if not messages and not tools:
            return 0

        self.warm_up()
        client = self.client
        if client is None:
            msg = "The Amazon Bedrock client was not initialized."
            raise RuntimeError(msg)

        system_prompts, bedrock_messages = _format_messages(messages)
        converse_input: dict[str, Any] = {"messages": bedrock_messages}
        if system_prompts:
            converse_input["system"] = system_prompts

        tool_config = _format_tools(flatten_tools_or_toolsets(tools)) if tools else None
        if tool_config:
            converse_input["toolConfig"] = tool_config

        try:
            response = client.count_tokens(modelId=self.model, input={"converse": converse_input})
        except ClientError as exception:
            msg = (
                f"Could not count tokens using the model '{self.model}'. Amazon Bedrock's CountTokens API sizes only a "
                f"complete, valid conversation and requires a model that supports token counting; see the underlying "
                f"error for details."
            )
            raise AmazonBedrockInferenceError(msg) from exception

        return response["inputTokens"]

    def close(self) -> None:
        """Close the Amazon Bedrock client and release its resources."""
        if self.client is not None:
            self.client.close()
            self.client = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the counter.

        :returns: A dictionary representation of the counter.
        """
        return default_to_dict(
            self,
            model=self.model,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            aws_region_name=self.aws_region_name,
            aws_profile_name=self.aws_profile_name,
            boto3_config=self.boto3_config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmazonBedrockTokenCounter":
        """
        Deserialize the counter.

        :param data: A dictionary representation of the counter.
        :returns: The deserialized counter.
        """
        return default_from_dict(cls, data)
