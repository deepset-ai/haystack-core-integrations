# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from google.genai import Client, types
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
from haystack.tools import ToolsType
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.common.google_genai.utils import _get_client
from haystack_integrations.components.generators.google_genai.chat.utils import (
    _convert_message_to_google_genai_format,
    _convert_tools_to_google_genai_format,
)


class GoogleGenAITokenCounter:
    """
    Counts input tokens for Gemini models with Google's token counting API.

    Unlike local token counters, this counter sends the input to the `countTokens` endpoint of the Google Gen AI
    SDK, so the returned count includes the model-specific formatting Gemini applies to messages.

    Inputs are assembled exactly as `GoogleGenAIChatGenerator` sends them: a leading system message becomes the
    system instruction and the remaining messages become the request contents.

    ### Backend support for system instructions and tools

    The Google Gen AI SDK only accepts a system instruction and tool schemas on `countTokens` when the client
    targets Vertex AI. On the Gemini Developer API it rejects both, so this counter raises a `ValueError` instead
    of silently returning a count that omits them. Counting plain messages works on either backend.

    ## Usage Example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.token_counters.google_genai import GoogleGenAITokenCounter

    counter = GoogleGenAITokenCounter("gemini-3.7-flash")
    messages = [ChatMessage.from_user("Hello, how are you?")]
    token_count = counter.count(messages)
    print(f"Token count: {token_count}")
    ```
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: str | None = None,
        vertex_ai_location: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Initialize the counter.

        :param model: The model whose tokenization should be used. Token counts are model-specific, so count
            against the same model you intend to generate with.
        :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment
            variables. Not needed if using Vertex AI with Application Default Credentials.
        :param api: Which API to use. Either `gemini` for the Gemini Developer API or `vertex` for Vertex AI.
        :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
            Application Default Credentials.
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., `us-central1`, `europe-west1`).
            Required when using Vertex AI with Application Default Credentials.
        :param timeout: Timeout for Google Gen AI client calls. If not set, it defaults to the default set by the
            Google Gen AI client.
        :param max_retries: Maximum number of retries to attempt for failed requests. If not set, it defaults to
            the default set by the Google Gen AI client.
        """
        self.model = model
        self.api_key = api_key
        self.api = api
        self.vertex_ai_project = vertex_ai_project
        self.vertex_ai_location = vertex_ai_location
        self.timeout = timeout
        self.max_retries = max_retries

        self.client: Client | None = None

    def warm_up(self) -> None:
        """Initialize the Google Gen AI client."""
        if self.client is not None:
            return

        self.client = _get_client(
            api_key=self.api_key,
            api=self.api,
            vertex_ai_project=self.vertex_ai_project,
            vertex_ai_location=self.vertex_ai_location,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Return the number of input tokens Gemini will use for the given messages and tools.

        :param messages: The messages to measure. A leading system message is measured as the system instruction.
        :param tools: Tools whose schemas are sent alongside the messages, and so consume tokens too.
        :returns: The token count, or `0` when there is nothing to measure.
        :raises ValueError: If a system message or tools are passed while targeting the Gemini Developer API,
            which cannot measure either.
        """
        if not messages and not tools:
            return 0

        # Mirror how GoogleGenAIChatGenerator splits the request: a leading system message is sent separately.
        chat_messages = messages
        system_instruction = None
        if messages and messages[0].is_from(ChatRole.SYSTEM):
            system_instruction = messages[0].text or ""
            chat_messages = messages[1:]

        config_params: dict[str, Any] = {}
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        if tools:
            config_params["tools"] = _convert_tools_to_google_genai_format(tools)

        # Rejected before the client is built, so an unsupported request fails the same way with or without
        # credentials.
        if config_params and self.api != "vertex":
            unsupported = " and ".join(
                name for name, present in (("a system message", system_instruction), ("tools", tools)) if present
            )
            msg = (
                f"Counting tokens for {unsupported} is only supported when targeting Vertex AI, because the "
                "Google Gen AI SDK rejects them on the Gemini Developer API. Initialize this counter with "
                'api="vertex", or count only the non-system messages.'
            )
            raise ValueError(msg)

        self.warm_up()
        client = self.client
        if client is None:
            msg = "The Google Gen AI client was not initialized."
            raise RuntimeError(msg)

        contents: list[types.ContentUnion] = [_convert_message_to_google_genai_format(msg) for msg in chat_messages]
        config = types.CountTokensConfig(**config_params) if config_params else None

        response = client.models.count_tokens(model=self.model, contents=contents, config=config)
        return response.total_tokens or 0

    def close(self) -> None:
        """Close the Google Gen AI client and its underlying HTTP resources."""
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
            api_key=self.api_key.to_dict(),
            api=self.api,
            vertex_ai_project=self.vertex_ai_project,
            vertex_ai_location=self.vertex_ai_location,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleGenAITokenCounter":
        """
        Deserialize the counter.

        :param data: The dictionary to deserialize from.
        :returns: The deserialized counter.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
