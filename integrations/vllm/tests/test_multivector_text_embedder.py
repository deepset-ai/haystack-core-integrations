# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import math
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.embedders.vllm import VLLMMultivectorTextEmbedder

MODEL = os.environ.get("VLLM_MULTIVECTOR_MODEL", "answerdotai/answerai-colbert-small-v1")
API_BASE_URL = "http://localhost:8003"
MODULE = "haystack_integrations.components.embedders.vllm.multivector_text_embedder"


def _fake_response(embeddings: list[list[list[float]]], *, status_code: int = 200):
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://localhost:8000/pooling"),
        json={
            "data": [{"index": i, "data": embedding} for i, embedding in enumerate(embeddings)],
            "model": "fake-model",
            "usage": {"prompt_tokens": len(embeddings), "total_tokens": len(embeddings)},
        },
    )


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)

        assert embedder.model == MODEL
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.api_base_url == "http://localhost:8000"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout is None
        assert embedder.http_client_kwargs is None
        assert embedder.extra_parameters is None
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_with_parameters(self):
        embedder = VLLMMultivectorTextEmbedder(
            model="custom-model",
            api_key=Secret.from_token("test-api-key"),
            api_base_url="https://vllm.example.test/",
            prefix="query: ",
            suffix="!",
            timeout=12.5,
            http_client_kwargs={"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
            extra_parameters={"truncate_prompt_tokens": 128, "use_activation": True},
        )

        assert embedder.model == "custom-model"
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "https://vllm.example.test"
        assert embedder.prefix == "query: "
        assert embedder.suffix == "!"
        assert embedder.timeout == 12.5
        assert embedder.http_client_kwargs == {
            "timeout": 99,
            "headers": {"X-Custom": "value"},
            "verify": False,
        }
        assert embedder.extra_parameters == {"truncate_prompt_tokens": 128, "use_activation": True}

    def test_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "secret-value")
        init_parameters = {
            "model": "custom-model",
            "api_key": Secret.from_env_var("CUSTOM_VLLM_KEY"),
            "api_base_url": "https://vllm.example.test",
            "prefix": "query: ",
            "suffix": "!",
            "timeout": 12.5,
            "http_client_kwargs": {"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
            "extra_parameters": {"truncate_prompt_tokens": 128, "use_activation": True},
        }
        embedder = VLLMMultivectorTextEmbedder(**init_parameters)

        serialized = component_to_dict(embedder, "embedder")
        assert serialized == {
            "type": f"{MODULE}.VLLMMultivectorTextEmbedder",
            "init_parameters": {
                **init_parameters,
                "api_key": {"type": "env_var", "env_vars": ["CUSTOM_VLLM_KEY"], "strict": True},
            },
        }
        restored = component_from_dict(VLLMMultivectorTextEmbedder, serialized, "embedder")
        assert component_to_dict(restored, "embedder") == serialized
        assert restored.api_key.resolve_value() == "secret-value"


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_API_KEY", raising=False)
        embedder = VLLMMultivectorTextEmbedder(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_API_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_API_KEY"):
            embedder.warm_up()

    @patch(f"{MODULE}.init_http_client")
    def test_sync_lifecycle(self, init_client, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "token")
        embedder = VLLMMultivectorTextEmbedder(
            model=MODEL,
            api_key=Secret.from_env_var("CUSTOM_VLLM_KEY"),
            timeout=2.5,
            http_client_kwargs={"timeout": 50, "headers": {"X-Custom": "yes"}, "verify": False},
        )
        client = MagicMock(spec=httpx.Client)
        init_client.return_value = client
        init_client.assert_not_called()

        embedder.warm_up()
        embedder.warm_up()
        init_client.assert_called_once_with(
            async_client=False,
            http_client_kwargs={
                "timeout": 2.5,
                "headers": httpx.Headers({"X-Custom": "yes", "Authorization": "Bearer token"}),
                "verify": False,
            },
        )
        assert embedder._client is client
        assert embedder._async_client is None
        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.warm_up()
        assert init_client.call_count == 2
        assert embedder._client is client

    @patch(f"{MODULE}.init_http_client")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, init_client):
        embedder = VLLMMultivectorTextEmbedder(
            model=MODEL, api_key=None, http_client_kwargs={"timeout": 7, "headers": {"Accept": "x"}}
        )
        client = MagicMock(spec=httpx.AsyncClient)
        init_client.return_value = client
        init_client.assert_not_called()

        await embedder.warm_up_async()
        await embedder.warm_up_async()
        init_client.assert_called_once_with(
            async_client=True,
            http_client_kwargs={"timeout": 7, "headers": httpx.Headers({"Accept": "x"})},
        )
        assert embedder._async_client is client
        assert embedder._client is None
        await embedder.close_async()
        client.aclose.assert_awaited_once_with()
        assert embedder._async_client is None
        await embedder.warm_up_async()
        assert init_client.call_count == 2
        assert embedder._async_client is client

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

        sync_client = MagicMock(spec=httpx.Client)
        async_client = MagicMock(spec=httpx.AsyncClient)
        embedder._client = sync_client
        embedder._async_client = async_client

        embedder.close()
        assert embedder._client is None
        assert embedder._async_client is async_client
        async_client.aclose.assert_not_awaited()
        await embedder.close_async()
        assert embedder._async_client is None
        embedder.close()
        await embedder.close_async()
        sync_client.close.assert_called_once_with()
        async_client.aclose.assert_awaited_once_with()


class TestRun:
    def test_run_wrong_input_format(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)
        with patch.object(embedder, "warm_up") as warm_up:
            with pytest.raises(TypeError, match="expects a string as input"):
                embedder.run(["not", "a string"])
        warm_up.assert_not_called()

    def test_run_with_mock(self):
        embedder = VLLMMultivectorTextEmbedder(
            model=MODEL,
            api_base_url="http://server.test/",
            prefix="[",
            suffix="]",
            extra_parameters={"model": "wrong", "input": "wrong", "task": "wrong", "truncate_prompt_tokens": 64},
        )
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = _fake_response([[[1.0, 2.5], [3.0, 4.0]]])

        result = embedder.run("hello")

        embedder._client.post.assert_called_once_with(
            "http://server.test/pooling",
            json={"truncate_prompt_tokens": 64, "model": MODEL, "input": "[hello]", "task": "token_embed"},
        )
        assert result == {"multivector_embedding": [[1.0, 2.5], [3.0, 4.0]]}

    @pytest.mark.asyncio
    async def test_run_async(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL, prefix="query: ")
        embedder._async_client = MagicMock(spec=httpx.AsyncClient)
        response = _fake_response([[[0.7, 0.8]]])
        embedder._async_client.post = AsyncMock(return_value=response)

        result = await embedder.run_async("rome")

        embedder._async_client.post.assert_awaited_once_with(
            "http://localhost:8000/pooling",
            json={"model": MODEL, "input": "query: rome", "task": "token_embed"},
        )
        assert result == {"multivector_embedding": [[0.7, 0.8]]}

    def test_run_raises_on_http_error(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)
        embedder._client = MagicMock(spec=httpx.Client)
        response = _fake_response([], status_code=503)
        embedder._client.post.return_value = response

        with pytest.raises(httpx.HTTPStatusError):
            embedder.run("hello")


class TestIntegration:
    @pytest.mark.integration
    def test_live_run(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL, api_base_url=API_BASE_URL, timeout=60)
        try:
            result = embedder.run("The food was delicious")
        finally:
            embedder.close()

        embedding = result["multivector_embedding"]
        assert embedding and embedding[0]
        assert all(len(vector) == len(embedding[0]) for vector in embedding)
        assert all(isinstance(value, float) and math.isfinite(value) for vector in embedding for value in vector)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL, api_base_url=API_BASE_URL, timeout=60)
        try:
            result = await embedder.run_async("The food was delicious")
        finally:
            await embedder.close_async()

        embedding = result["multivector_embedding"]
        assert embedding and embedding[0]
        assert all(len(vector) == len(embedding[0]) for vector in embedding)
        assert all(isinstance(value, float) and math.isfinite(value) for vector in embedding for value in vector)
