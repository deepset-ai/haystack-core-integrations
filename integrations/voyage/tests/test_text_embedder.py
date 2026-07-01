# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.voyage import VoyageTextEmbedder


class TestVoyageTextEmbedder:
    def test_supported_models(self):
        models = VoyageTextEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageTextEmbedder()
        assert embedder.api_key == Secret.from_env_var("VOYAGE_API_KEY")
        assert embedder.model == "voyage-3.5"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.input_type == "query"
        assert embedder.truncation is True
        assert embedder.output_dimension is None
        assert embedder.output_dtype is None
        assert embedder.timeout is None

    def test_init_with_parameters(self):
        embedder = VoyageTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="voyage-3-large",
            prefix="pre ",
            suffix=" post",
            input_type="document",
            truncation=False,
            output_dimension=512,
            output_dtype="int8",
            timeout=60.0,
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "voyage-3-large"
        assert embedder.prefix == "pre "
        assert embedder.suffix == " post"
        assert embedder.input_type == "document"
        assert embedder.truncation is False
        assert embedder.output_dimension == 512
        assert embedder.output_dtype == "int8"
        assert embedder.timeout == 60.0

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        component_dict = VoyageTextEmbedder().to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.voyage.text_embedder.VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3.5",
                "prefix": "",
                "suffix": "",
                "input_type": "query",
                "truncation": True,
                "output_dimension": None,
                "output_dtype": None,
                "timeout": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage.text_embedder.VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3-large",
                "prefix": "",
                "suffix": "",
                "input_type": "query",
                "truncation": True,
                "output_dimension": 256,
                "output_dtype": None,
                "timeout": None,
            },
        }
        embedder = VoyageTextEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var("VOYAGE_API_KEY")
        assert embedder.model == "voyage-3-large"
        assert embedder.output_dimension == 256

    def test_run_wrong_input_format(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageTextEmbedder()
        with pytest.raises(TypeError):
            embedder.run(text=["I'm a list, not a string"])

    def test_run(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageTextEmbedder(prefix="pre ", suffix=" post")
        embedder._client.embed = MagicMock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]], total_tokens=4))

        result = embedder.run(text="I love pizza!")

        embedder._client.embed.assert_called_once_with(
            texts=["pre I love pizza! post"],
            model="voyage-3.5",
            input_type="query",
            truncation=True,
            output_dimension=None,
            output_dtype=None,
        )
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["meta"] == {"model": "voyage-3.5", "total_tokens": 4}

    @pytest.mark.asyncio
    async def test_run_async(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageTextEmbedder()
        embedder._async_client.embed = AsyncMock(
            return_value=SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]], total_tokens=4)
        )

        result = await embedder.run_async(text="I love pizza!")

        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["meta"] == {"model": "voyage-3.5", "total_tokens": 4}

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    def test_live_run(self):
        embedder = VoyageTextEmbedder()
        result = embedder.run(text="The food was delicious")
        assert len(result["embedding"]) > 0
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    async def test_live_run_async(self):
        embedder = VoyageTextEmbedder()
        result = await embedder.run_async(text="The food was delicious")
        assert len(result["embedding"]) > 0
        assert result["meta"]["total_tokens"] > 0
