# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, ClassVar

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

DEFAULT_API_BASE_URL = "https://api.hpc-ai.com/inference/v1"


@component
class HPCAIChatGenerator(OpenAIChatGenerator):
    """
    Enables chat completion using HPC-AI's OpenAI-compatible API.

    HPC-AI exposes an OpenAI-compatible `/chat/completions` endpoint, so this component
    reuses the standard OpenAI chat payload shape and response handling from Haystack's
    `OpenAIChatGenerator`.

    You can pass any chat completion parameters supported by HPC-AI directly through
    `generation_kwargs` during initialization or when calling `run`.
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "minimax/minimax-m2.5",
        "moonshotai/kimi-k2.5",
    ]
    """The HPC-AI models officially supported by this integration."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("HPC_AI_API_KEY"),
        model: str = "minimax/minimax-m2.5",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of `HPCAIChatGenerator`.

        :param api_key:
            The HPC-AI API key.
        :param model:
            The HPC-AI chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param api_base_url:
            The HPC-AI API base URL. If not provided, the component reads `HPC_AI_BASE_URL`
            and falls back to `https://api.hpc-ai.com/inference/v1`.
        :param generation_kwargs:
            Additional parameters forwarded directly to the HPC-AI chat completion endpoint.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls.
        :param timeout:
            The timeout for the HPC-AI API call.
        :param extra_headers:
            Additional HTTP headers to include in requests to the HPC-AI API.
        :param max_retries:
            Maximum number of retries to contact HPC-AI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5.
        :param http_client_kwargs:
            Keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
        """
        api_base_url = api_base_url or os.getenv("HPC_AI_BASE_URL", DEFAULT_API_BASE_URL)

        super(HPCAIChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            tools=tools,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )
        self.extra_headers = extra_headers

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=serialize_tools_or_toolset(self.tools),
            extra_headers=self.extra_headers,
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
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        extra_headers = {**(self.extra_headers or {})}

        openai_formatted_messages: list[dict[str, Any]] = [message.to_openai_dict_format() for message in messages]

        tools_strict = tools_strict if tools_strict is not None else self.tools_strict
        flattened_tools = flatten_tools_or_toolsets(tools or self.tools)
        _check_duplicate_tool_names(flattened_tools)

        openai_tools = {}
        if flattened_tools:
            tool_definitions = []
            for tool in flattened_tools:
                function_spec = {**tool.tool_spec}
                if tools_strict:
                    function_spec["strict"] = True
                    parameters = function_spec.get("parameters")
                    if isinstance(parameters, dict):
                        parameters["additionalProperties"] = False
                tool_definitions.append({"type": "function", "function": function_spec})
            openai_tools = {"tools": tool_definitions}

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)

        if is_streaming and num_responses > 1:
            msg = "Cannot stream multiple responses, please set n=1."
            raise ValueError(msg)

        response_format = generation_kwargs.pop("response_format", None)
        base_args = {
            "model": self.model,
            "messages": openai_formatted_messages,
            "n": num_responses,
            **openai_tools,
            "extra_body": {**generation_kwargs},
            "extra_headers": {**extra_headers},
        }
        if response_format and not is_streaming:
            return {**base_args, "response_format": response_format, "openai_endpoint": "parse"}

        final_args = {**base_args, "stream": is_streaming, "openai_endpoint": "create"}
        if response_format:
            final_args["response_format"] = response_format
        return final_args
