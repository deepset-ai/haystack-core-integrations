# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.utils import Secret
from haystack.utils.http_client import init_http_client
from openai import AsyncOpenAI, OpenAI


def _create_openai_clients(
    api_key: Secret | None,
    api_base_url: str,
    timeout: float | None,
    max_retries: int | None,
    http_client_kwargs: dict[str, Any] | None,
) -> tuple[OpenAI, AsyncOpenAI]:
    """
    Build sync and async OpenAI clients pointing at a vLLM server.

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

    sync_client = OpenAI(http_client=init_http_client(http_client_kwargs, async_client=False), **client_kwargs)
    async_client = AsyncOpenAI(http_client=init_http_client(http_client_kwargs, async_client=True), **client_kwargs)
    return sync_client, async_client
