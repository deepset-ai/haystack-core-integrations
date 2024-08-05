# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret
from haystack_integrations.components.embedders.jina import JinaTextEmbedder


class TestJinaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaTextEmbedder()

        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")
        assert embedder.model_name == "jina-embeddings-v2-base-en"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = JinaTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model_name == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

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
                "model": "jina-embeddings-v2-base-en",
                "prefix": "",
                "suffix": "",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaTextEmbedder(
            model="model",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "model",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    def test_run(self):
        model = "jina-embeddings-v2-base-en"
        with patch("requests.sessions.Session.post") as mock_post:
            # Configure the mock to return a specific response
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps(
                {
                    "model": "jina-embeddings-v2-base-en",
                    "object": "list",
                    "usage": {"total_tokens": 6, "prompt_tokens": 6},
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                }
            ).encode()

            mock_post.return_value = mock_response

            embedder = JinaTextEmbedder(
                api_key=Secret.from_token("fake-api-key"), model=model, prefix="prefix ", suffix=" suffix"
            )
            result = embedder.run(text="The food was delicious")

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
