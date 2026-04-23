# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import httpx
import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.jina import JinaTextEmbedder


class TestJinaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaTextEmbedder()

        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")
        assert embedder.model_name == "jina-embeddings-v3"
        assert embedder.base_url == "https://api.jina.ai/v1/embeddings"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = JinaTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            base_url="https://my.custom.url/v1/embeddings",
            prefix="prefix",
            suffix="suffix",
            late_chunking=True,
        )
        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model_name == "model"
        assert embedder.base_url == "https://my.custom.url/v1/embeddings"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.late_chunking is True

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            JinaTextEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaTextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-embeddings-v3",
                "base_url": "https://api.jina.ai/v1/embeddings",
                "prefix": "",
                "suffix": "",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "model",
                "base_url": "https://my.custom.url/v1/embeddings",
                "prefix": "prefix",
                "suffix": "suffix",
                "task": "retrieval.query",
                "dimensions": 1024,
                "late_chunking": True,
            },
        }
        embedder = JinaTextEmbedder.from_dict(data)

        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")
        assert embedder.model_name == "model"
        assert embedder.base_url == "https://my.custom.url/v1/embeddings"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.task == "retrieval.query"
        assert embedder.dimensions == 1024
        assert embedder.late_chunking is True

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaTextEmbedder(
            model="model",
            base_url="https://my.custom.url/v1/embeddings",
            prefix="prefix",
            suffix="suffix",
            task="retrieval.query",
            dimensions=1024,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "model",
                "base_url": "https://my.custom.url/v1/embeddings",
                "prefix": "prefix",
                "suffix": "suffix",
                "task": "retrieval.query",
                "dimensions": 1024,
            },
        }

    def test_run(self):
        model = "jina-embeddings-v2-base-en"
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v2-base-en",
                "object": "list",
                "usage": {"total_tokens": 6, "prompt_tokens": 6},
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            },
        )

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            embedder = JinaTextEmbedder(
                api_key=Secret.from_token("fake-api-key"), model=model, prefix="prefix ", suffix=" suffix"
            )
            result = embedder.run(text="The food was delicious")

        mock_post.assert_called_once()
        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "model": "jina-embeddings-v2-base-en",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }

    def test_run_wrong_input_format(self):
        embedder = JinaTextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="JinaTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    def test_with_v3(self):
        model = "jina-embeddings-v3"
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v3",
                "object": "list",
                "usage": {"total_tokens": 6, "prompt_tokens": 6},
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            },
        )

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            embedder = JinaTextEmbedder(
                api_key=Secret.from_token("fake-api-key"),
                model=model,
                prefix="prefix ",
                suffix=" suffix",
                task="retrieval.query",
            )
            result = embedder.run(text="The food was delicious")

        mock_post.assert_called_once()
        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "model": "jina-embeddings-v3",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }

    @pytest.mark.asyncio
    async def test_run_async(self):
        model = "jina-embeddings-v2-base-en"
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v2-base-en",
                "object": "list",
                "usage": {"total_tokens": 6, "prompt_tokens": 6},
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            },
        )

        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            embedder = JinaTextEmbedder(
                api_key=Secret.from_token("fake-api-key"), model=model, prefix="prefix ", suffix=" suffix"
            )
            result = await embedder.run_async(text="The food was delicious")

        mock_post.assert_called_once()
        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "model": "jina-embeddings-v2-base-en",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }

    @pytest.mark.asyncio
    async def test_run_async_wrong_input_format(self):
        embedder = JinaTextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="JinaTextEmbedder expects a string as an input"):
            await embedder.run_async(text=list_integers_input)

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY env var not set")
    @pytest.mark.integration
    def test_run_integration(self):
        embedder = JinaTextEmbedder(task="retrieval.query")
        result = embedder.run(text="What is the capital of France?")

        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
        assert all(isinstance(x, (int, float)) for x in result["embedding"])

        assert "meta" in result
        assert isinstance(result["meta"], dict)
        assert "model" in result["meta"]
        assert "usage" in result["meta"]

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY env var not set")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        embedder = JinaTextEmbedder(task="retrieval.query")
        result = await embedder.run_async(text="What is the capital of France?")

        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
        assert all(isinstance(x, (int, float)) for x in result["embedding"])

        assert "meta" in result
        assert isinstance(result["meta"], dict)
        assert "model" in result["meta"]
        assert "usage" in result["meta"]
