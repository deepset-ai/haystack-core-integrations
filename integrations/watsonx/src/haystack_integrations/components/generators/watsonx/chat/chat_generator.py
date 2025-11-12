# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Any, Literal, get_args

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    ImageContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    TextContent,
    select_streaming_callback,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

logger = logging.getLogger(__name__)


# See https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-prompt-data.html?context=wx
# for supported formats
ImageFormat = Literal["image/jpeg", "image/png"]
IMAGE_SUPPORTED_FORMATS: list[ImageFormat] = list(get_args(ImageFormat))


@component
class WatsonxChatGenerator:
    """
    Enables chat completions using IBM's watsonx.ai foundation models.

    This component interacts with IBM's watsonx.ai platform to generate chat responses using various foundation
    models. It supports the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format for both input
    and output, including multimodal inputs with text and images.

    The generator works with IBM's foundation models that are listed
    [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&audience=wdp).

    You can customize the generation behavior by passing parameters to the watsonx.ai API through the
    `generation_kwargs` argument. These parameters are passed directly to the watsonx.ai inference endpoint.

    For details on watsonx.ai API parameters, see
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    messages = [ChatMessage.from_user("Explain quantum computing in simple terms")]

    client = WatsonxChatGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        model="ibm/granite-13b-chat-v2",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )
    response = client.run(messages)
    print(response)
    ```

    ### Multimodal usage example

    ```python
    from haystack.dataclasses import ChatMessage, ImageContent

    # Create an image from file path or base64
    image_content = ImageContent.from_file_path("path/to/your/image.jpg")

    # Create a multimodal message with both text and image
    messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

    # Use a multimodal model
    client = WatsonxChatGenerator(
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        model="meta-llama/llama-3-2-11b-vision-instruct",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )
    response = client.run(messages)
    print(response)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),  # noqa: B008
        model: str = "ibm/granite-3-3-8b-instruct",
        project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),  # noqa: B008
        api_base_url: str = "https://us-south.ml.cloud.ibm.com",
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        verify: bool | str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> None:
        """
        Creates an instance of WatsonxChatGenerator.

        Before initializing the component, you can set environment variables:
        - `WATSONX_TIMEOUT` to override the default timeout
        - `WATSONX_MAX_RETRIES` to override the default retry count

        :param api_key: IBM Cloud API key for watsonx.ai access.
            Can be set via `WATSONX_API_KEY` environment variable or passed directly.
        :param model: The model ID to use for completions. Defaults to "ibm/granite-13b-chat-v2".
            Available models can be found in your IBM Cloud account.
        :param project_id: IBM Cloud project ID
        :param api_base_url: Custom base URL for the API endpoint.
            Defaults to "https://us-south.ml.cloud.ibm.com".
        :param generation_kwargs: Additional parameters to control text generation.
            These parameters are passed directly to the watsonx.ai inference endpoint.
            Supported parameters include:
            - `temperature`: Controls randomness (lower = more deterministic)
            - `max_new_tokens`: Maximum number of tokens to generate
            - `min_new_tokens`: Minimum number of tokens to generate
            - `top_p`: Nucleus sampling probability threshold
            - `top_k`: Number of highest probability tokens to consider
            - `repetition_penalty`: Penalty for repeated tokens
            - `length_penalty`: Penalty based on output length
            - `stop_sequences`: List of sequences where generation should stop
            - `random_seed`: Seed for reproducible results
        :param timeout: Timeout in seconds for API requests.
            Defaults to environment variable `WATSONX_TIMEOUT` or 30 seconds.
        :param max_retries: Maximum number of retry attempts for failed requests.
            Defaults to environment variable `WATSONX_MAX_RETRIES` or 5.
        :param verify: SSL verification setting. Can be:
            - True: Verify SSL certificates (default)
            - False: Skip verification (insecure)
            - Path to CA bundle for custom certificates
        :param streaming_callback: A callback function for streaming responses.
        """
        self.api_key = api_key
        self.model = model
        self.project_id = project_id
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify = verify
        self.streaming_callback = streaming_callback

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Watsonx client with configured credentials."""
        credentials = Credentials(api_key=self.api_key.resolve_value(), url=self.api_base_url)

        self.client = ModelInference(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id.resolve_value(),
            verify=self.verify,
            max_retries=self.max_retries,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            project_id=self.project_id.to_dict(),
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            verify=self.verify,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatsonxChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "project_id"])
        init_params = data.get("init_parameters", {})
        serialized_callback = init_params.get("streaming_callback")
        if serialized_callback:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback)
        return default_from_dict(cls, data)

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        *,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Generate chat completions synchronously.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided this will override the `streaming_callback` set in the `__init__` method.
        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if not messages:
            return {"replies": []}

        resolved_streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        api_args = self._prepare_api_call(messages=messages, generation_kwargs=generation_kwargs)

        if resolved_streaming_callback:
            return self._handle_streaming(api_args=api_args, callback=resolved_streaming_callback)

        return self._handle_standard(api_args)

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        *,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Generate chat completions asynchronously.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            If provided this will override the `streaming_callback` set in the `__init__` method.
        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if not messages:
            return {"replies": []}

        resolved_streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        api_args = self._prepare_api_call(messages=messages, generation_kwargs=generation_kwargs)

        if resolved_streaming_callback:
            return await self._handle_async_streaming(api_args=api_args, callback=resolved_streaming_callback)

        return await self._handle_async_standard(api_args)

    def _prepare_api_call(
        self, *, messages: list[ChatMessage], generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        watsonx_messages = []
        content: str | None | dict[str, Any] | list[dict[str, Any]]

        for msg in messages:
            if msg.is_from("tool"):
                logger.debug("Skipping tool message - tool calls are not currently supported")
                continue

            # Check that images are only in user messages
            if msg.images and not msg.is_from(ChatRole.USER):
                error_msg = "Image content is only supported for user messages"
                raise ValueError(error_msg)

            # Handle multimodal content (text + images) preserving order
            if msg.images:
                # Pre-validate all images first (following LlamaCpp pattern)
                for image in msg.images:
                    if image.mime_type not in IMAGE_SUPPORTED_FORMATS:
                        supported_formats = ", ".join(IMAGE_SUPPORTED_FORMATS)
                        msg_error = (
                            f"Unsupported image format: {image.mime_type}. "
                            f"WatsonX supports the following formats: {supported_formats}"
                        )
                        raise ValueError(msg_error)

                content_parts: list[dict[str, Any]] = []
                for part in msg._content:
                    if isinstance(part, TextContent) and part.text:
                        content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        # WatsonX expects base64 data URI format
                        # See: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-api-chat.html?context=wx
                        image_url = f"data:{part.mime_type};base64,{part.base64_image}"
                        content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

                content = content_parts
            else:
                # Simple text-only message
                content = msg.text

            watsonx_msg = {"role": msg.role.value, "content": content}
            if msg.name:
                watsonx_msg["name"] = msg.name
            watsonx_messages.append(watsonx_msg)

        merged_kwargs.pop("stream", None)

        return {"messages": watsonx_messages, "params": merged_kwargs}

    def _handle_streaming(
        self,
        *,
        api_args: dict[str, Any],
        callback: SyncStreamingCallbackT,
    ) -> dict[str, list[ChatMessage]]:
        """
        Handle synchronous streaming response.

        :param api_args: Arguments for the API call, including messages and parameters.
        :param callback: A callback function to handle streaming chunks.
        :returns:
            A dictionary with the generated responses as ChatMessage instances.
        """
        chunks: list[StreamingChunk] = []
        stream = self.client.chat_stream(messages=api_args["messages"], params=api_args["params"])

        for chunk in stream:
            if not isinstance(chunk, dict) or not chunk.get("choices"):
                continue

            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                chunk_meta = {
                    "model": self.model,
                    "index": chunk["choices"][0].get("index", 0),
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
                streaming_chunk = StreamingChunk(content=content, meta=chunk_meta)
                chunks.append(streaming_chunk)
                callback(streaming_chunk)

        return {"replies": [self._convert_streaming_chunks_to_chat_message(chunks)]}

    def _handle_standard(self, api_args: dict[str, Any]) -> dict[str, list[ChatMessage]]:
        """Handle synchronous standard response."""
        response = self.client.chat(messages=api_args["messages"], params=api_args["params"])
        return self._process_response(response)

    async def _handle_async_streaming(
        self,
        *,
        api_args: dict[str, Any],
        callback: AsyncStreamingCallbackT,
    ) -> dict[str, list[ChatMessage]]:
        """Handle asynchronous streaming response."""
        chunks: list[StreamingChunk] = []
        stream_generator = await self.client.achat_stream(messages=api_args["messages"], params=api_args["params"])

        async for chunk in stream_generator:
            if not isinstance(chunk, dict) or not chunk.get("choices"):
                continue

            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                chunk_meta = {
                    "model": self.model,
                    "index": chunk["choices"][0].get("index", 0),
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
                streaming_chunk = StreamingChunk(content=content, meta=chunk_meta)
                chunks.append(streaming_chunk)
                await callback(streaming_chunk)

        return {"replies": [self._convert_streaming_chunks_to_chat_message(chunks)]}

    def _convert_streaming_chunks_to_chat_message(self, chunks: list[StreamingChunk]) -> ChatMessage:
        """Convert list of streaming chunks to a single ChatMessage."""
        if not chunks:
            return ChatMessage.from_assistant("")

        content = "".join(chunk.content for chunk in chunks)
        last_chunk_meta = chunks[-1].meta if chunks else {}

        return ChatMessage.from_assistant(
            text=content,
            meta={
                "model": self.model,
                "finish_reason": last_chunk_meta.get("finish_reason"),
                "usage": last_chunk_meta.get("usage", {}),
                "chunks_count": len(chunks),
            },
        )

    async def _handle_async_standard(self, api_args: dict[str, Any]) -> dict[str, list[ChatMessage]]:
        """Handle asynchronous standard response."""
        response = await self.client.achat(messages=api_args["messages"], params=api_args["params"])
        return self._process_response(response)

    def _process_response(self, response: dict[str, Any]) -> dict[str, list[ChatMessage]]:
        """Process standard response into Haystack format."""
        if not response.get("choices"):
            return {"replies": []}

        choice = response["choices"][0]
        message = choice.get("message", {})
        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=message.get("content", ""),
                    meta={
                        "model": self.model,
                        "index": choice.get("index", 0),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": response.get("usage", {}),
                    },
                )
            ]
        }
