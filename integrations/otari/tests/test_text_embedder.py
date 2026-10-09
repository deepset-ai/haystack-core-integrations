# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from haystack.utils import Secret
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.otari import OtariTextEmbedder

from .conftest import OTARI_API_BASE_URL, requires_api_key

DEFAULT_MODEL = "openai:text-embedding-3-small"
DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"
COMPONENT_TYPE = "haystack_integrations.components.embedders.otari.text_embedder.OtariTextEmbedder"


class TestOtariTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariTextEmbedder()
        assert embedder.api_key == Secret.from_env_var("OTARI_API_KEY")
        assert embedder.model == DEFAULT_MODEL
        assert embedder.dimensions is None
        assert embedder.api_base_url == DEFAULT_API_BASE_URL
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        assert OtariTextEmbedder().to_dict() == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "api_key": {"env_vars": ["OTARI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "dimensions": None,
                "api_base_url": DEFAULT_API_BASE_URL,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_from_dict_round_trip_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        embedder = OtariTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="mistral:mistral-embed",
            dimensions=256,
            api_base_url="https://otari.example.com/api/v1",
            prefix="query: ",
            suffix=" end",
            timeout=10,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = embedder.to_dict()
        assert "organization" not in data["init_parameters"]

        restored = OtariTextEmbedder.from_dict(data)
        assert restored.api_key == Secret.from_env_var("ENV_VAR")
        assert restored.model == "mistral:mistral-embed"
        assert restored.dimensions == 256
        assert restored.api_base_url == "https://otari.example.com/api/v1"
        assert restored.prefix == "query: "
        assert restored.suffix == " end"
        assert restored.timeout == 10
        assert restored.max_retries == 2
        assert restored.http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_run(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariTextEmbedder(dimensions=3, prefix="query: ")
        with patch("openai.resources.embeddings.Embeddings.create") as mock_create:
            mock_create.return_value = CreateEmbeddingResponse(
                data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
                model="text-embedding-3-small",
                object="list",
                usage=Usage(prompt_tokens=4, total_tokens=4),
            )
            result = embedder.run("I love pizza!")

        _, kwargs = mock_create.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["input"] == "query: I love pizza!"
        assert kwargs["dimensions"] == 3
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["meta"] == {
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    @requires_api_key
    @pytest.mark.integration
    def test_live_run(self):
        embedder = OtariTextEmbedder(api_base_url=OTARI_API_BASE_URL, dimensions=256)
        result = embedder.run("I love pizza!")

        assert len(result["embedding"]) == 256
        assert all(isinstance(x, float) for x in result["embedding"])
        assert "text-embedding-3-small" in result["meta"]["model"]
