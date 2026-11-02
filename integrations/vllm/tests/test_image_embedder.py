# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ImageContent
from haystack.utils import Secret
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.vllm import VLLMImageEmbedder

MODEL = "TIGER-Lab/VLM2Vec-Full"
# 1x1 transparent PNG
BASE64_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def _fake_response(embedding: list[float]) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model="fake-model",
        data=[Embedding(object="embedding", index=0, embedding=embedding)],
        usage=Usage(prompt_tokens=4, total_tokens=4),
    )


@pytest.fixture
def image() -> ImageContent:
    return ImageContent(base64_image=BASE64_PNG, mime_type="image/png")


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMImageEmbedder(model=MODEL)
        assert embedder.model == MODEL
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.instruction is None
        assert embedder.dimensions is None
        assert embedder.extra_parameters is None
        assert embedder._client is None
        assert embedder._async_client is None

    def test_to_dict_from_dict_roundtrip_with_non_default_params(self, monkeypatch):
        monkeypatch.setenv("MY_VLLM_KEY", "secret-value")
        embedder = VLLMImageEmbedder(
            model=MODEL,
            api_key=Secret.from_env_var("MY_VLLM_KEY"),
            api_base_url="http://my-vllm-server:8000/v1",
            instruction="Represent the given image.",
            dimensions=128,
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
            extra_parameters={"truncate_prompt_tokens": 256},
        )
        data = component_to_dict(embedder, "embedder")
        assert data["type"] == "haystack_integrations.components.embedders.vllm.image_embedder.VLLMImageEmbedder"

        restored = component_from_dict(VLLMImageEmbedder, data, "embedder")
        assert restored.api_key.resolve_value() == "secret-value"
        for attr in (
            "model",
            "api_base_url",
            "instruction",
            "dimensions",
            "timeout",
            "max_retries",
            "http_client_kwargs",
            "extra_parameters",
        ):
            assert getattr(restored, attr) == getattr(embedder, attr)


class TestRun:
    def test_run(self, image):
        embedder = VLLMImageEmbedder(model=MODEL, instruction="Represent the given image.", dimensions=64)
        embedder._client = MagicMock()
        embedder._client.post.return_value = _fake_response([0.1, 0.2, 0.3])

        result = embedder.run(image=image)

        assert result == {
            "embedding": [0.1, 0.2, 0.3],
            "meta": {"model": "fake-model", "usage": {"prompt_tokens": 4, "total_tokens": 4}},
        }
        call = embedder._client.post.call_args
        assert call.args == ("/embeddings",)
        assert call.kwargs["cast_to"] is CreateEmbeddingResponse
        assert call.kwargs["body"] == {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{BASE64_PNG}"}},
                        {"type": "text", "text": "Represent the given image."},
                    ],
                }
            ],
            "encoding_format": "float",
            "dimensions": 64,
        }

    def test_run_wrong_input_type(self):
        embedder = VLLMImageEmbedder(model=MODEL)
        with pytest.raises(TypeError, match="VLLMDocumentImageEmbedder"):
            embedder.run(image="not an image")

    @pytest.mark.asyncio
    async def test_run_async(self, image):
        embedder = VLLMImageEmbedder(model=MODEL)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(return_value=_fake_response([0.4, 0.5]))

        result = await embedder.run_async(image=image)

        assert result["embedding"] == [0.4, 0.5]
        content = embedder._async_client.post.call_args.kwargs["body"]["messages"][0]["content"]
        # no instruction configured, so only the image is sent
        assert content == [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{BASE64_PNG}"}}]

    @pytest.mark.asyncio
    async def test_run_async_wrong_input_type(self):
        embedder = VLLMImageEmbedder(model=MODEL)
        with pytest.raises(TypeError, match="VLLMDocumentImageEmbedder"):
            await embedder.run_async(image="not an image")
