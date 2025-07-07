# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from google.genai.types import ContentEmbedding, EmbedContentConfig, EmbedContentResponse
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder


class TestGoogleGenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAITextEmbedder()

        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model_name == "text-embedding-004"
        assert embedder._prefix == ""
        assert embedder._suffix == ""
        assert embedder._config == {"task_type": "SEMANTIC_SIMILARITY"}
        assert embedder._api == "gemini"
        assert embedder._vertex_ai_project is None
        assert embedder._vertex_ai_location is None

    def test_init_with_parameters(self):
        embedder = GoogleGenAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            prefix="prefix",
            suffix="suffix",
            config={"task_type": "CLASSIFICATION"},
        )
        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model_name == "model"
        assert embedder._prefix == "prefix"
        assert embedder._suffix == "suffix"
        assert embedder._config == {"task_type": "CLASSIFICATION"}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        component = GoogleGenAITextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "model": "text-embedding-004",
                "prefix": "",
                "suffix": "",
                "config": {"task_type": "SEMANTIC_SIMILARITY"},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = GoogleGenAITextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            prefix="prefix",
            suffix="suffix",
            config={"task_type": "CLASSIFICATION"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder",
            "init_parameters": {
                "model": "model",
                "api_key": {"type": "env_var", "env_vars": ["ENV_VAR"], "strict": False},
                "prefix": "prefix",
                "suffix": "suffix",
                "config": {"task_type": "CLASSIFICATION"},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "model": "text-embedding-004",
                "prefix": "",
                "suffix": "",
                "config": {"task_type": "CLASSIFICATION"},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
            },
        }
        component = GoogleGenAITextEmbedder.from_dict(data)
        assert component._api_key.resolve_value() == "fake-api-key"
        assert component._model_name == "text-embedding-004"
        assert component._prefix == ""
        assert component._suffix == ""
        assert component._config == {"task_type": "CLASSIFICATION"}

    def test_prepare_input(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAITextEmbedder()

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
                auto_truncate=None,
            ),
        }

    def test_prepare_output(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")

        response = EmbedContentResponse(
            embeddings=[ContentEmbedding(values=[0.1, 0.2, 0.3])],
        )

        embedder = GoogleGenAITextEmbedder()
        result = embedder._prepare_output(result=response)
        assert result == {
            "embedding": [0.1, 0.2, 0.3],
            "meta": {"model": "text-embedding-004"},
        }

    def test_run_wrong_input_format(self):
        embedder = GoogleGenAITextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="GoogleGenAITextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        model = "text-embedding-004"

        embedder = GoogleGenAITextEmbedder(model=model)
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 768
        assert all(isinstance(x, float) for x in result["embedding"])

        assert "text" in result["meta"]["model"] and "004" in result["meta"]["model"], (
            "The model name does not contain 'text' and '004'"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    async def test_run_async(self):
        model = "text-embedding-004"

        embedder = GoogleGenAITextEmbedder(model=model)
        result = await embedder.run_async(text="The food was delicious")

        assert len(result["embedding"]) == 768
        assert all(isinstance(x, float) for x in result["embedding"])

        assert "text" in result["meta"]["model"] and "004" in result["meta"]["model"], (
            "The model name does not contain 'text' and '004'"
        )
        assert result["meta"] == {"model": model}
