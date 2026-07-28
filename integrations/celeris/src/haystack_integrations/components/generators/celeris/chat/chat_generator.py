# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import ToolsType, flatten_tools_or_toolsets, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

CELERIS_DEFAULT_MODEL = "celeris-1"
CELERIS_DEFAULT_API_BASE_URL = "https://inference.celeris.ai/celeris-1/v1"

#: Prompt and completion share a single 8192-token budget on Celeris.
CELERIS_CONTEXT_LIMIT = 8192
#: `max_tokens` must be 1 (warm ping) or a positive multiple of 256.
CELERIS_MAX_TOKENS_MULTIPLE = 256
#: Celeris leaves little prompt room by default, so we always send an explicit, quantized value.
CELERIS_DEFAULT_MAX_TOKENS = 1024
#: The single non-multiple-of-256 value Celeris accepts; it is meant for warming a connection.
CELERIS_WARM_PING_MAX_TOKENS = 1

#: Generation parameters Celeris does not implement. Passing them through results in a 400.
CELERIS_UNSUPPORTED_GENERATION_KWARGS = {
    "response_format": (
        "Celeris does not support the 'response_format' parameter: it offers neither JSON mode nor JSON-schema "
        "structured output. Route structured output through tool calling instead."
    ),
}

# Conservative characters-per-token ratio used to estimate how much of the shared context the prompt
# consumes. Celeris does not publish a tokenizer, so we deliberately over-estimate: over-estimating
# shrinks `max_tokens` (shorter answer), under-estimating would produce a 400 from the API.
_CHARS_PER_TOKEN = 3
# Per-message envelope (role, delimiters) charged on top of the message content.
_PER_MESSAGE_TOKEN_OVERHEAD = 4


def _message_character_count(message: ChatMessage) -> int:
    """
    Count the characters a `ChatMessage` contributes to the prompt.

    :param message: The message to measure.
    :returns: The number of characters carried by the message.
    """
    num_characters = sum(len(text) for text in message.texts)
    for tool_call in message.tool_calls:
        num_characters += len(tool_call.tool_name) + len(json.dumps(tool_call.arguments))
    for tool_call_result in message.tool_call_results:
        num_characters += len(tool_call_result.result)
    if message.name:
        num_characters += len(message.name)
    return num_characters


@component
class CelerisChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Celeris.

    Celeris serves general-purpose diffusion-based language models over an OpenAI-compatible API. Rather than
    generating one token at a time, they produce tokens in parallel blocks, which makes them substantially faster
    than autoregressive models of comparable quality. The speed advantage is most visible in latency-sensitive
    work.

    The component talks to the Celeris chat completions endpoint and uses the ChatMessage format for both input
    and output; see the [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage) for details.

    Celeris has a few hard API constraints that this component enforces for you:

    - **`max_tokens` must be `1` or a positive multiple of 256.** The value you pass is rounded *up* to the next
      multiple of 256. If you pass nothing, `1024` is used.
    - **Prompt and completion share a single 8192-token context budget.** `max_tokens` is additionally capped
      (down, to a multiple of 256) so that the estimated prompt plus `max_tokens` fits in that budget. If the
      prompt leaves less than 256 tokens of room, a `ValueError` is raised instead of letting the API return a 400.
    - **`response_format` is not supported.** Celeris offers neither JSON mode nor JSON-schema
      structured output; route structured output through tool calling instead. Passing it raises a `ValueError`.
    - **`tool_choice="required"` is not supported.** Use `"auto"`, `"none"`, or a named tool.
    - **Input and output are text only.** Messages carrying images raise a `ValueError`.

    Streaming and tool calling are supported. Note that Celeris generates in blocks, so streamed chunks arrive in
    large groups rather than one at a time.

    Usage example:
    ```python
    from haystack_integrations.components.generators.celeris import CelerisChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = CelerisChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("CELERIS_API_KEY"),
        model: str = CELERIS_DEFAULT_MODEL,
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = CELERIS_DEFAULT_API_BASE_URL,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool = False,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of CelerisChatGenerator.

        :param api_key:
            The Celeris API key. Read from the `CELERIS_API_KEY` environment variable by default.
        :param model:
            The name of the Celeris chat completion model to use. `celeris-1` is the only model served today.
            Note that Celeris encodes the model in the endpoint path, so changing this also requires changing
            `api_base_url`.
        :param streaming_callback:
            A callback function that is called when a new chunk is received from the stream.
            The callback function accepts a StreamingChunk as an argument.
        :param api_base_url:
            The Celeris API base URL. The default (`https://inference.celeris.ai/celeris-1/v1`) is pinned to the
            default model, because Celeris serves each model under its own path.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are sent directly to the Celeris endpoint.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have. Rounded up to the next multiple
                of 256 and capped so that prompt + `max_tokens` stays within the 8192-token shared context.
                Defaults to `1024`. Pass `1` for a warm ping.
            - `temperature`: The sampling temperature to use. Higher values mean the model takes more risks.
            - `top_p`: The nucleus sampling value to use.
            - `seed`: The seed to use for sampling.
            - `stop`: One or more sequences at which generation stops.
            - `n`: How many completions to generate.
            - `presence_penalty` / `frequency_penalty`: Penalties applied while sampling.
            - `logprobs` / `top_logprobs`: Returned per-token log probabilities. The Celeris API reference lists
                `logprobs` as unsupported, but the endpoint accepts it and returns `logprobs` on each choice, so
                this component passes it through rather than rejecting it.
            `response_format` is not supported by Celeris and raises a `ValueError`.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance. `tool_choice="required"` is not supported by Celeris.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model follows exactly
            the schema provided in the `parameters` field of the tool definition.
        :param timeout:
            The timeout for the Celeris API call.
        :param max_retries:
            Maximum number of retries to contact Celeris after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(CelerisChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

        if model != CELERIS_DEFAULT_MODEL and api_base_url == CELERIS_DEFAULT_API_BASE_URL:
            logger.warning(
                "Celeris serves each model under its own endpoint path, so 'api_base_url' is pinned to the "
                "default model '{default_model}'. You set model='{model}' but left 'api_base_url' at "
                "'{api_base_url}', which will keep routing requests to '{default_model}'. Set 'api_base_url' "
                "to the path of the model you want to call.",
                default_model=CELERIS_DEFAULT_MODEL,
                model=model,
                api_base_url=api_base_url,
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            tools=serialize_tools_or_toolset(self.tools),
            tools_strict=self.tools_strict,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    def _prepare_api_call(
        self,
        *,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, Any]:
        # merge exactly the way the parent does, so validation sees the effective request
        merged_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        self._check_unsupported_generation_kwargs(merged_generation_kwargs)
        self._check_text_only(messages)

        resolved_tools = flatten_tools_or_toolsets(tools or self.tools)
        prompt_tokens = self._estimate_prompt_tokens(messages=messages, tools=resolved_tools)
        merged_generation_kwargs["max_tokens"] = self._resolve_max_tokens(
            requested_max_tokens=merged_generation_kwargs.get("max_tokens"), prompt_tokens=prompt_tokens
        )

        # `@component` rebuilds the class, so the zero-argument form of `super()` cannot be used here
        return super(CelerisChatGenerator, self)._prepare_api_call(  # noqa: UP008
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=merged_generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )

    @staticmethod
    def _check_unsupported_generation_kwargs(generation_kwargs: dict[str, Any]) -> None:
        """
        Reject generation parameters that Celeris does not implement.

        :param generation_kwargs: The effective generation parameters for this request.
        :raises ValueError: If an unsupported parameter is present.
        """
        for unsupported, msg in CELERIS_UNSUPPORTED_GENERATION_KWARGS.items():
            if generation_kwargs.get(unsupported) is not None:
                raise ValueError(msg)

        if generation_kwargs.get("tool_choice") == "required":
            msg = (
                "Celeris does not support tool_choice='required'. Use 'auto', 'none', or a named tool "
                "(for example {'type': 'function', 'function': {'name': 'my_tool'}})."
            )
            raise ValueError(msg)

    @staticmethod
    def _check_text_only(messages: list[ChatMessage]) -> None:
        """
        Reject messages carrying non-text content.

        :param messages: The messages that make up the prompt.
        :raises ValueError: If any message contains an image.
        """
        for index, message in enumerate(messages):
            if message.images:
                msg = (
                    f"Celeris is a text-only model: image input is not supported, but message at index {index} "
                    f"contains {len(message.images)} image(s)."
                )
                raise ValueError(msg)

    @staticmethod
    def _estimate_prompt_tokens(*, messages: list[ChatMessage], tools: list[Any]) -> int:
        """
        Estimate how much of the shared 8192-token context the prompt consumes.

        Celeris does not publish a tokenizer, so this is a deliberately conservative character-based estimate:
        over-estimating only shortens the completion, while under-estimating would make the API reject the request.

        :param messages: The messages that make up the prompt.
        :param tools: The tool definitions sent along with the prompt.
        :returns: The estimated number of prompt tokens.
        """
        num_characters = sum(_message_character_count(message) for message in messages)
        for tool in tools:
            num_characters += len(json.dumps(tool.tool_spec))

        overhead = _PER_MESSAGE_TOKEN_OVERHEAD * (len(messages) + len(tools))
        return -(-num_characters // _CHARS_PER_TOKEN) + overhead

    @staticmethod
    def _resolve_max_tokens(*, requested_max_tokens: int | None, prompt_tokens: int) -> int:
        """
        Quantize `max_tokens` to Celeris' 256-token grid and fit it into the shared context budget.

        :param requested_max_tokens: The value the caller asked for, if any.
        :param prompt_tokens: The estimated number of prompt tokens.
        :returns: A value Celeris accepts: `1`, or a positive multiple of 256 that fits the remaining budget.
        :raises ValueError: If `max_tokens` is not a positive integer, or if the prompt leaves no room for a
            completion.
        """
        if requested_max_tokens is None:
            requested_max_tokens = CELERIS_DEFAULT_MAX_TOKENS

        if not isinstance(requested_max_tokens, int) or isinstance(requested_max_tokens, bool):
            msg = f"'max_tokens' must be an int, got {type(requested_max_tokens).__name__}."
            raise ValueError(msg)
        if requested_max_tokens < 1:
            msg = f"'max_tokens' must be a positive integer, got {requested_max_tokens}."
            raise ValueError(msg)

        # 1 is the warm-ping value Celeris accepts as-is; it always fits.
        if requested_max_tokens == CELERIS_WARM_PING_MAX_TOKENS:
            return CELERIS_WARM_PING_MAX_TOKENS

        # round UP to the next multiple of 256
        quantized = -(-requested_max_tokens // CELERIS_MAX_TOKENS_MULTIPLE) * CELERIS_MAX_TOKENS_MULTIPLE

        # prompt and completion share one budget: floor the remaining room to a multiple of 256
        remaining = CELERIS_CONTEXT_LIMIT - prompt_tokens
        affordable = (remaining // CELERIS_MAX_TOKENS_MULTIPLE) * CELERIS_MAX_TOKENS_MULTIPLE
        if affordable < CELERIS_MAX_TOKENS_MULTIPLE:
            msg = (
                f"The prompt is estimated at {prompt_tokens} tokens, which leaves fewer than "
                f"{CELERIS_MAX_TOKENS_MULTIPLE} tokens of Celeris' {CELERIS_CONTEXT_LIMIT}-token context for the "
                f"completion. Prompt and completion share one budget on Celeris, and 'max_tokens' must be a "
                f"positive multiple of {CELERIS_MAX_TOKENS_MULTIPLE}. Shorten the prompt."
            )
            raise ValueError(msg)

        if quantized > affordable:
            logger.debug(
                "Capping 'max_tokens' from {quantized} to {affordable}: the prompt is estimated at "
                "{prompt_tokens} tokens and Celeris shares a {context_limit}-token budget between prompt and "
                "completion.",
                quantized=quantized,
                affordable=affordable,
                prompt_tokens=prompt_tokens,
                context_limit=CELERIS_CONTEXT_LIMIT,
            )

        return min(quantized, affordable)
