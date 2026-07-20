# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.edenai.text_embedder import EdenAITextEmbedder

API_BASE_URL = "https://api.edenai.run/v3"
DEFAULT_MODEL = "openai/text-embedding-3-small"


class TestEdenAITextEmbedder:
    def test_supported_models(self):
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = EdenAITextEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")

        embedder = EdenAITextEmbedder()
        assert embedder.api_key == Secret.from_env_var(["EDENAI_API_KEY"])
        assert embedder.api_base_url == API_BASE_URL
        assert embedder.model == DEFAULT_MODEL
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("EDENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            EdenAITextEmbedder()

    def test_init_with_parameters(self):
        embedder = EdenAITextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="mistral/mistral-embed",
            prefix="START",
            suffix="END",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == API_BASE_URL
        assert embedder.model == "mistral/mistral-embed"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")

        embedder_component = EdenAITextEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.edenai.text_embedder.EdenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["EDENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "api_base_url": API_BASE_URL,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = EdenAITextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="mistral/mistral-embed",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
        )
        component_dict = embedder.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.edenai.text_embedder.EdenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "mistral/mistral-embed",
                "api_base_url": "https://custom-api-base-url.com",
                "prefix": "START",
                "suffix": "END",
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "https://proxy.example.com"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-secret-key")
        data = {
            "type": "haystack_integrations.components.embedders.edenai.text_embedder.EdenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["EDENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "api_base_url": API_BASE_URL,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        embedder = EdenAITextEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var(["EDENAI_API_KEY"])
        assert embedder.api_base_url == API_BASE_URL
        assert embedder.model == DEFAULT_MODEL
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None

    def test_run_wrong_input_format(self):
        embedder = EdenAITextEmbedder(api_key=Secret.from_token("test-api-key"))
        list_integers_input = ["text_snippet_1", "text_snippet_2"]
        match_error_msg = (
            "OpenAITextEmbedder expects a string as an input.In case you want to embed a list of Documents,"
            " please use the OpenAIDocumentEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY", None),
        reason="Export an env var called EDENAI_API_KEY containing the Eden AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = EdenAITextEmbedder()
        result = embedder.run("The food was delicious")
        assert len(result["embedding"]) > 0
        assert all(isinstance(x, float) for x in result["embedding"])
