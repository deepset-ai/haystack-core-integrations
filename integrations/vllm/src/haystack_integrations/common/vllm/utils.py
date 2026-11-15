# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.utils import Secret
from haystack.utils.http_client import init_http_client
from openai import AsyncOpenAI, OpenAI


def _openai_client_kwargs(
    api_key: Secret | None,
    api_base_url: str,
    timeout: float | None,
    max_retries: int | None,
) -> dict[str, Any]:
    """
    Build the common keyword arguments for OpenAI clients pointing at a vLLM server.

    A placeholder api key is used when the user did not supply one and no `VLLM_API_KEY` env var is set, because the
    OpenAI client requires a non-empty value.
    `timeout` and `max_retries` are only forwarded when provided: when None, the OpenAI client's own defaults apply.
    """
    resolved_api_key = "placeholder-api-key"
    if api_key is not None and (value := api_key.resolve_value()):
        resolved_api_key = value

    client_kwargs: dict[str, Any] = {"api_key": resolved_api_key, "base_url": api_base_url}
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries

    return client_kwargs


def _create_openai_client(
    api_key: Secret | None,
    api_base_url: str,
    timeout: float | None,
    max_retries: int | None,
    http_client_kwargs: dict[str, Any] | None,
) -> OpenAI:
    """Build a synchronous OpenAI client pointing at a vLLM server."""
    client_kwargs = _openai_client_kwargs(api_key, api_base_url, timeout, max_retries)
    sync_http_client = init_http_client(http_client_kwargs, async_client=False)
    # openai>=3 annotates http_client as httpx2, but legacy httpx clients are supported at runtime.
    # https://github.com/openai/openai-python/blob/main/httpx2.md
    return OpenAI(http_client=sync_http_client, **client_kwargs)  # type: ignore[arg-type]


def _create_async_openai_client(
    api_key: Secret | None,
    api_base_url: str,
    timeout: float | None,
    max_retries: int | None,
    http_client_kwargs: dict[str, Any] | None,
) -> AsyncOpenAI:
    """Build an asynchronous OpenAI client pointing at a vLLM server."""
    client_kwargs = _openai_client_kwargs(api_key, api_base_url, timeout, max_retries)
    async_http_client = init_http_client(http_client_kwargs, async_client=True)
    # openai>=3 annotates http_client as httpx2, but legacy httpx clients are supported at runtime.
    # https://github.com/openai/openai-python/blob/main/httpx2.md
    return AsyncOpenAI(http_client=async_http_client, **client_kwargs)  # type: ignore[arg-type]


def _prepare_image_embedding_body(
    *,
    model: str,
    image_url: str,
    instruction: str | None,
    dimensions: int | None,
    extra_parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build the request body to embed one image with the vLLM `/embeddings` endpoint.

    vLLM accepts chat-style `messages` instead of `input` for multimodal embedding models. The whole conversation is
    treated as a single prompt, so each request embeds exactly one image.
    See the [vLLM docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/).
    """
    content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": image_url}}]
    if instruction:
        content.append({"type": "text", "text": instruction})

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "encoding_format": "float",
    }
    if dimensions is not None:
        body["dimensions"] = dimensions
    if extra_parameters:
        body.update(extra_parameters)
    return body
