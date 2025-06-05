# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from google.genai import types
from google.genai.types import ContentEmbedding, EmbedContentConfig, EmbedContentResponse
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.google_genai import GoogleAITextEmbedder


class TestGoogleAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleAITextEmbedder()

        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model_name == "text-embedding-004"

    def test_init_with_parameters(self):
        embedder = GoogleAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
        )
        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model_name == "model"

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        embedder = GoogleAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model_name == "model"
        assert embedder._config == types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY")

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            GoogleAITextEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        component = GoogleAITextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.google_genai.text_embedder.GoogleAITextEmbedder",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY"], "strict": True},
                "model": "text-embedding-004",
                "config": {"task_type": "SEMANTIC_SIMILARITY"}
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = GoogleAITextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY"
            )
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.google_genai.text_embedder.GoogleAITextEmbedder",
            "init_parameters": {
                "model": "model",
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["ENV_VAR"],
                    "strict": False
                },
                "config": {"task_type": "SEMANTIC_SIMILARITY"}
            }
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.google_genai.text_embedder.GoogleAITextEmbedder",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY"], "strict": True},
                "model": "text-embedding-004",
            },
        }
        component = GoogleAITextEmbedder.from_dict(data)
        assert component._api_key.resolve_value() == "fake-api-key"
        assert component._model_name == "text-embedding-004"

    def test_prepare_input(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleAITextEmbedder()

        contents = "The food was delicious"
        prepared_input = embedder._prepare_input(contents)
        assert prepared_input == {
            "model": "text-embedding-004",
            "contents": "The food was delicious",
            "config": EmbedContentConfig(
                http_options=None,
                task_type="SEMANTIC_SIMILARITY",
                title=None,
                output_dimensionality=None,
                mime_type=None,
                auto_truncate=None
            )
        }

    def test_prepare_output(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")

        response = EmbedContentResponse(
            embeddings=[ContentEmbedding(values=[0.1, 0.2, 0.3])],
        )

        embedder = GoogleAITextEmbedder()
        result = embedder._prepare_output(result=response)
        assert result == {
            "embedding": [0.1, 0.2, 0.3],
            "meta": {"model": "text-embedding-004"},
        }

    def test_run_wrong_input_format(self):
        embedder = GoogleAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="GoogleAITextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(os.environ.get("GOOGLE_API_KEY", "") == "", reason="GOOGLE_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        model = "text-embedding-004"

        embedder = GoogleAITextEmbedder(model=model)
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 768
        assert all(isinstance(x, float) for x in result["embedding"])

        assert "text" in result["meta"]["model"] and "004" in result["meta"]["model"], (
            "The model name does not contain 'text' and '004'"
        )
