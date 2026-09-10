# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils import Secret

from haystack_integrations.common.vllm.utils import _create_async_openai_client, _create_openai_client


def test_create_openai_client_placeholder_when_no_key():
    client = _create_openai_client(
        api_key=None, api_base_url="http://localhost:8000/v1", timeout=None, max_retries=None, http_client_kwargs=None
    )
    assert client.api_key == "placeholder-api-key"
    assert str(client.base_url) == "http://localhost:8000/v1/"
    client.close()


def test_create_openai_client_uses_resolved_key_and_forwards_options():
    client = _create_openai_client(
        api_key=Secret.from_token("real-key"),
        api_base_url="http://vllm:8000/v1",
        timeout=12.5,
        max_retries=7,
        http_client_kwargs=None,
    )
    assert client.api_key == "real-key"
    assert client.timeout == 12.5
    assert client.max_retries == 7
    client.close()


@pytest.mark.asyncio
async def test_create_async_openai_client_placeholder_when_no_key():
    client = _create_async_openai_client(
        api_key=None, api_base_url="http://localhost:8000/v1", timeout=None, max_retries=None, http_client_kwargs=None
    )
    assert client.api_key == "placeholder-api-key"
    assert str(client.base_url) == "http://localhost:8000/v1/"
    await client.close()
