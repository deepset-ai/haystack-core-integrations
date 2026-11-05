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
from haystack_integrations.components.embedders.vllm.multivector_text_embedder import (
    VLLMMultivectorTextEmbedder as DirectVLLMMultivectorTextEmbedder,
)

MODEL = "jinaai/jina-colbert-v2"
MODULE = "haystack_integrations.components.embedders.vllm.multivector_text_embedder"


def _response(payload, *, status_code=200):
    request = httpx.Request("POST", "http://localhost:8000/pooling")
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "request failed", request=request, response=httpx.Response(status_code, request=request)
        )
    return response


def _payload(index=0, vectors=None, *, model="served-model", usage=None):
    return {
        "data": [{"index": index, "data": vectors or [[0.1, 0.2], [0.3, 0.4]]}],
        "model": model,
        "usage": usage or {"prompt_tokens": 3, "total_tokens": 3},
    }


class TestInitializationAndSerialization:
    def test_default_constructor_state_and_export(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)

        assert DirectVLLMMultivectorTextEmbedder is VLLMMultivectorTextEmbedder
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
        assert set(embedder.run._output_types_cache) == {"multivector_embedding"}
        assert embedder.run._output_types_cache["multivector_embedding"].type == list[list[float]]
        assert set(embedder.run_async._output_types_cache) == {"multivector_embedding"}
        assert embedder.run_async._output_types_cache["multivector_embedding"].type == list[list[float]]

    def test_non_default_constructor_and_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "secret-value")
        embedder = VLLMMultivectorTextEmbedder(
            model="custom-model",
            api_key=Secret.from_env_var("CUSTOM_VLLM_KEY"),
            api_base_url="https://vllm.example.test/",
            prefix="query: ",
            suffix="!",
            timeout=12.5,
            http_client_kwargs={"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
            extra_parameters={"truncate_prompt_tokens": 128, "normalize": True},
        )

        serialized = component_to_dict(embedder, "embedder")
        assert serialized == {
            "type": f"{MODULE}.VLLMMultivectorTextEmbedder",
            "init_parameters": {
                "model": "custom-model",
                "api_key": {"type": "env_var", "env_vars": ["CUSTOM_VLLM_KEY"], "strict": True},
                "api_base_url": "https://vllm.example.test",
                "prefix": "query: ",
                "suffix": "!",
                "timeout": 12.5,
                "http_client_kwargs": {"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
                "extra_parameters": {"truncate_prompt_tokens": 128, "normalize": True},
            },
        }

        restored = component_from_dict(VLLMMultivectorTextEmbedder, serialized, "embedder")
        assert restored.model == "custom-model"
        assert restored.api_key == Secret.from_env_var("CUSTOM_VLLM_KEY")
        assert restored.api_key.resolve_value() == "secret-value"
        assert restored.api_base_url == "https://vllm.example.test"
        assert restored.prefix == "query: "
        assert restored.suffix == "!"
        assert restored.timeout == 12.5
        assert restored.http_client_kwargs == {
            "timeout": 99,
            "headers": {"X-Custom": "value"},
            "verify": False,
        }
        assert restored.extra_parameters == {"truncate_prompt_tokens": 128, "normalize": True}


class TestComponentLifecycle:
    def test_sync_warm_up_is_lazy_idempotent_and_builds_client_options(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "token")
        client = MagicMock(spec=httpx.Client)
        with patch(f"{MODULE}.init_http_client", return_value=client) as init_client:
            embedder = VLLMMultivectorTextEmbedder(
                model=MODEL,
                api_key=Secret.from_env_var("CUSTOM_VLLM_KEY"),
                timeout=2.5,
                http_client_kwargs={"timeout": 50, "headers": {"X-Custom": "yes"}, "verify": False},
            )
            init_client.assert_not_called()

            embedder.warm_up()
            embedder.warm_up()

        init_client.assert_called_once()
        kwargs = init_client.call_args.kwargs
        assert kwargs["async_client"] is False
        assert kwargs["http_client_kwargs"]["timeout"] == 2.5
        assert kwargs["http_client_kwargs"]["verify"] is False
        headers = kwargs["http_client_kwargs"]["headers"]
        assert headers["Authorization"] == "Bearer token"
        assert headers["X-Custom"] == "yes"
        assert embedder._client is client

        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.close()

    @pytest.mark.asyncio
    async def test_async_warm_up_is_lazy_idempotent_and_close_is_independent(self):
        async_client = MagicMock(spec=httpx.AsyncClient)
        async_client.aclose = AsyncMock()
        with patch(f"{MODULE}.init_http_client", return_value=async_client) as init_client:
            embedder = VLLMMultivectorTextEmbedder(
                model=MODEL, api_key=None, http_client_kwargs={"timeout": 7, "headers": {"Accept": "x"}}
            )
            await embedder.warm_up_async()
            await embedder.warm_up_async()

        init_client.assert_called_once()
        kwargs = init_client.call_args.kwargs
        assert kwargs["async_client"] is True
        assert kwargs["http_client_kwargs"]["timeout"] == 7
        assert "Authorization" not in kwargs["http_client_kwargs"]["headers"]
        assert kwargs["http_client_kwargs"]["headers"]["Accept"] == "x"
        assert embedder._client is None

        await embedder.close_async()
        async_client.aclose.assert_awaited_once_with()
        assert embedder._async_client is None
        await embedder.close_async()
        embedder.close()

    def test_secret_is_resolved_only_during_warm_up(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_KEY", raising=False)
        embedder = VLLMMultivectorTextEmbedder(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_KEY"):
            embedder.warm_up()


class TestRun:
    def test_prepare_input(self):
        embedder = VLLMMultivectorTextEmbedder(
            model=MODEL,
            prefix="[",
            suffix="]",
            extra_parameters={"model": "wrong", "input": "wrong", "task": "wrong", "dimensions": 64},
        )

        assert embedder._prepare_input("hello") == {
            "dimensions": 64,
            "model": MODEL,
            "input": "[hello]",
            "task": "token_embed",
        }

    def test_type_validation_happens_before_warm_up(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)
        with patch.object(embedder, "warm_up") as warm_up:
            with pytest.raises(TypeError, match="expects a string as input"):
                embedder.run(["not", "a string"])
        warm_up.assert_not_called()

    def test_payload_url_precedence_and_output(self):
        embedder = VLLMMultivectorTextEmbedder(
            model=MODEL,
            api_base_url="http://server.test/",
            prefix="[",
            suffix="]",
            extra_parameters={"model": "wrong", "input": "wrong", "task": "wrong", "dimensions": 64},
        )
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = _response(_payload(vectors=[[1.0, 2.5], [3.0, 4.0]]))

        result = embedder.run("hello")

        embedder._client.post.assert_called_once_with(
            "http://server.test/pooling",
            json={"dimensions": 64, "model": MODEL, "input": "[hello]", "task": "token_embed"},
        )
        embedder._client.post.return_value.raise_for_status.assert_called_once_with()
        assert result == {"multivector_embedding": [[1.0, 2.5], [3.0, 4.0]]}

    @pytest.mark.asyncio
    async def test_run_async_parity(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL, prefix="query: ")
        embedder._async_client = MagicMock(spec=httpx.AsyncClient)
        response = _response(_payload(vectors=[[0.7, 0.8]]))
        embedder._async_client.post = AsyncMock(return_value=response)

        result = await embedder.run_async("rome")

        embedder._async_client.post.assert_awaited_once_with(
            "http://localhost:8000/pooling",
            json={"model": MODEL, "input": "query: rome", "task": "token_embed"},
        )
        response.raise_for_status.assert_called_once_with()
        assert result == {"multivector_embedding": [[0.7, 0.8]]}

    def test_http_error_is_raised(self):
        embedder = VLLMMultivectorTextEmbedder(model=MODEL)
        embedder._client = MagicMock(spec=httpx.Client)
        response = _response({}, status_code=503)
        embedder._client.post.return_value = response

        with pytest.raises(httpx.HTTPStatusError):
            embedder.run("hello")

        embedder._client.post.assert_called_once_with(
            "http://localhost:8000/pooling",
            json={"model": MODEL, "input": "hello", "task": "token_embed"},
        )
        response.raise_for_status.assert_called_once_with()


@pytest.mark.integration
class TestIntegration:
    @staticmethod
    def _assert_multivector(multivector: list[list[float]]) -> None:
        assert multivector
        dimensions = {len(vector) for vector in multivector}
        assert len(dimensions) == 1
        assert dimensions.pop() > 0
        assert all(isinstance(value, float) and math.isfinite(value) for vector in multivector for value in vector)

    def test_run(self):
        embedder = VLLMMultivectorTextEmbedder(
            model=os.environ.get("VLLM_MULTIVECTOR_MODEL", "answerdotai/answerai-colbert-small-v1"),
            api_base_url="http://localhost:8003",
            timeout=60,
        )

        try:
            result = embedder.run("Which city is the capital of Italy?")

            assert set(result) == {"multivector_embedding"}
            self._assert_multivector(result["multivector_embedding"])
        finally:
            embedder.close()

    @pytest.mark.asyncio
    async def test_run_async(self):
        embedder = VLLMMultivectorTextEmbedder(
            model=os.environ.get("VLLM_MULTIVECTOR_MODEL", "answerdotai/answerai-colbert-small-v1"),
            api_base_url="http://localhost:8003",
            timeout=60,
        )

        try:
            result = await embedder.run_async("Which city is the capital of Italy?")

            assert set(result) == {"multivector_embedding"}
            self._assert_multivector(result["multivector_embedding"])
        finally:
            await embedder.close_async()
