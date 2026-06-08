# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.utils import Secret

from haystack_integrations.common.vllm.utils import _create_openai_clients


def test_create_openai_clients_placeholder_when_no_key():
    sync_client, async_client = _create_openai_clients(
        api_key=None, api_base_url="http://localhost:8000/v1", timeout=None, max_retries=None, http_client_kwargs=None
    )
    assert sync_client.api_key == "placeholder-api-key"
    assert async_client.api_key == "placeholder-api-key"
    assert str(sync_client.base_url) == "http://localhost:8000/v1/"


def test_create_openai_clients_uses_resolved_key_and_forwards_options():
    sync_client, _ = _create_openai_clients(
        api_key=Secret.from_token("real-key"),
        api_base_url="http://vllm:8000/v1",
        timeout=12.5,
        max_retries=7,
        http_client_kwargs=None,
    )
    assert sync_client.api_key == "real-key"
    assert sync_client.timeout == 12.5
    assert sync_client.max_retries == 7
