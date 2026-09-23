# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.vllm import VLLMTextEmbedder

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_BASE_URL = "http://localhost:8001/v1"


def _fake_response(embeddings: list[list[float]], prompt_tokens: int = 5, total_tokens: int = 5):
    return CreateEmbeddingResponse(
        object="list",
        model="fake-model",
        data=[Embedding(object="embedding", index=i, embedding=e) for i, e in enumerate(embeddings)],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMTextEmbedder(model=MODEL)
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.model == MODEL
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.dimensions is None
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None
        assert embedder.extra_parameters is None
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_with_parameters(self):
        embedder = VLLMTextEmbedder(
            model=MODEL,
            api_key=Secret.from_token("test-api-key"),
            api_base_url="http://my-vllm-server:8000/v1",
            prefix="START",
            suffix="END",
            dimensions=64,
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
            extra_parameters={"dimensions": 32, "truncate_prompt_tokens": 256},
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "http://my-vllm-server:8000/v1"
        assert embedder.model == MODEL
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.dimensions == 64
        assert embedder.timeout == 10.0
        assert embedder.max_retries == 2
        assert embedder.http_client_kwargs == {"proxy": "https://proxy.example.com"}
        assert embedder.extra_parameters == {"dimensions": 32, "truncate_prompt_tokens": 256}

    def test_to_dict(self):
        component_dict = component_to_dict(VLLMTextEmbedder(model=MODEL), "embedder")
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.vllm.text_embedder.VLLMTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"},
                "model": MODEL,
                "api_base_url": "http://localhost:8000/v1",
                "prefix": "",
                "suffix": "",
                "dimensions": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
                "extra_parameters": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.vllm.text_embedder.VLLMTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"},
                "model": MODEL,
                "api_base_url": "http://localhost:8000/v1",
                "prefix": "",
                "suffix": "",
                "dimensions": 32,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
                "extra_parameters": None,
            },
        }
        embedder = component_from_dict(VLLMTextEmbedder, data, "embedder")
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.model == MODEL
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.dimensions == 32
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None
        assert embedder.extra_parameters is None


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_API_KEY", raising=False)
        embedder = VLLMTextEmbedder(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_API_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_API_KEY"):
            embedder.warm_up()

    @patch("haystack_integrations.components.embedders.vllm.text_embedder._create_openai_client")
    def test_sync_lifecycle(self, mock_client_cls):
        embedder = VLLMTextEmbedder(model=MODEL)
        client = mock_client_cls.return_value

        embedder.warm_up()
        assert embedder._client is client
        assert embedder._async_client is None
        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.vllm.text_embedder._create_async_openai_client")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, mock_client_cls):
        embedder = VLLMTextEmbedder(model=MODEL)
        client = MagicMock(close=AsyncMock())
        mock_client_cls.return_value = client

        await embedder.warm_up_async()
        assert embedder._async_client is client
        assert embedder._client is None
        await embedder.close_async()
        client.close.assert_awaited_once_with()
        assert embedder._async_client is None
        await embedder.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.vllm.text_embedder._create_openai_client")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        embedder = VLLMTextEmbedder(model=MODEL)
        embedder.warm_up()
        embedder.warm_up()
        mock_client_cls.assert_called_once()

    @patch("haystack_integrations.components.embedders.vllm.text_embedder._create_async_openai_client")
    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        embedder = VLLMTextEmbedder(model=MODEL)
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_client_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        embedder = VLLMTextEmbedder(model=MODEL)
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        embedder = VLLMTextEmbedder(model=MODEL)
        sync_client = MagicMock()
        async_client = MagicMock(close=AsyncMock())
        embedder._client = sync_client
        embedder._async_client = async_client

        embedder.close()
        assert embedder._client is None
        assert embedder._async_client is async_client
        async_client.close.assert_not_awaited()
        await embedder.close_async()
        assert embedder._async_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    def test_prepare_input_adds_dimensions_and_extra_body(self):
        embedder = VLLMTextEmbedder(
            model=MODEL, prefix="[", suffix="]", dimensions=32, extra_parameters={"truncate_prompt_tokens": 256}
        )
        kwargs = embedder._prepare_input("hello")
        assert kwargs == {
            "model": MODEL,
            "input": "[hello]",
            "encoding_format": "float",
            "dimensions": 32,
            "extra_body": {"truncate_prompt_tokens": 256},
        }

    def test_run_wrong_input_format(self):
        embedder = VLLMTextEmbedder(model=MODEL)
        with pytest.raises(TypeError, match=r"VLLMTextEmbedder expects a string as an input\."):
            embedder.run(text=["text_1", "text_2"])

    def test_run_with_mock(self):
        embedder = VLLMTextEmbedder(model=MODEL, prefix="[", suffix="]", dimensions=2)
        embedder._client = MagicMock()
        embedder._client.embeddings.create.return_value = _fake_response([[0.1, 0.2]])

        result = embedder.run("hello")

        call_kwargs = embedder._client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == "[hello]"
        assert call_kwargs["dimensions"] == 2
        assert result == {
            "embedding": [0.1, 0.2],
            "meta": {"model": "fake-model", "usage": {"prompt_tokens": 5, "total_tokens": 5}},
        }

    @pytest.mark.asyncio
    async def test_run_async(self):
        embedder = VLLMTextEmbedder(model=MODEL)
        embedder._async_client = MagicMock()
        embedder._async_client.embeddings.create = AsyncMock(return_value=_fake_response([[0.3, 0.4]]))

        result = await embedder.run_async("world")
        assert result["embedding"] == [0.3, 0.4]


class TestIntegration:
    @pytest.mark.integration
    def test_live_run(self):
        embedder = VLLMTextEmbedder(model=MODEL, api_base_url=API_BASE_URL)
        result = embedder.run("The food was delicious")
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_live_run_async(self):
        embedder = VLLMTextEmbedder(model=MODEL, api_base_url=API_BASE_URL)
        result = await embedder.run_async("The food was delicious")
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
