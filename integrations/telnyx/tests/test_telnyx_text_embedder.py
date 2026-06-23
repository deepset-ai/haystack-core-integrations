# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.telnyx.text_embedder import TelnyxTextEmbedder


class TestTelnyxTextEmbedder:
    def test_supported_models(self):
        models = TelnyxTextEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")

        embedder = TelnyxTextEmbedder()
        assert embedder.api_key == Secret.from_env_var(["TELNYX_API_KEY"])
        assert embedder.api_base_url == "https://api.telnyx.com/v2/ai/openai"
        assert embedder.model == "thenlper/gte-large"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.dimensions is None

    def test_init_with_parameters(self):
        embedder = TelnyxTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="custom-dimensions-model",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            dimensions=256,
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.model == "custom-dimensions-model"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.dimensions == 256

    def test_init_rejects_unsupported_dimensions(self):
        with pytest.raises(ValueError, match="does not support custom dimensions"):
            TelnyxTextEmbedder(api_key=Secret.from_token("test-api-key"), dimensions=256)

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")

        component_dict = TelnyxTextEmbedder().to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.telnyx.text_embedder.TelnyxTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["TELNYX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "thenlper/gte-large",
                "api_base_url": "https://api.telnyx.com/v2/ai/openai",
                "prefix": "",
                "suffix": "",
                "dimensions": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = TelnyxTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="custom-dimensions-model",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            dimensions=256,
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
        )
        assert embedder.to_dict() == {
            "type": "haystack_integrations.components.embedders.telnyx.text_embedder.TelnyxTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "custom-dimensions-model",
                "api_base_url": "https://custom-api-base-url.com",
                "prefix": "START",
                "suffix": "END",
                "dimensions": 256,
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "https://proxy.example.com"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-secret-key")
        data = {
            "type": "haystack_integrations.components.embedders.telnyx.text_embedder.TelnyxTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["TELNYX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "thenlper/gte-large",
                "api_base_url": "https://api.telnyx.com/v2/ai/openai",
                "prefix": "",
                "suffix": "",
                "dimensions": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        embedder = TelnyxTextEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var(["TELNYX_API_KEY"])
        assert embedder.api_base_url == "https://api.telnyx.com/v2/ai/openai"
        assert embedder.model == "thenlper/gte-large"
        assert embedder.dimensions is None

    def test_run_wrong_input_format(self):
        embedder = TelnyxTextEmbedder(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(TypeError, match="OpenAITextEmbedder expects a string as an input"):
            embedder.run(text=["text_snippet_1", "text_snippet_2"])

    @pytest.mark.skipif(
        not os.environ.get("TELNYX_API_KEY", None),
        reason="Export TELNYX_API_KEY to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        result = TelnyxTextEmbedder().run("The food was delicious")
        assert all(isinstance(value, float) for value in result["embedding"])
