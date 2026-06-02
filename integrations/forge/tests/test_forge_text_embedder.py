# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.forge.text_embedder import ForgeTextEmbedder


class TestForgeTextEmbedder:
    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS is a non-empty list of strings that includes the default model."""
        models = ForgeTextEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)
        assert "forge-pro" in models

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "test-api-key")

        embedder = ForgeTextEmbedder()
        assert embedder.api_key == Secret.from_env_var(["FORGE_API_KEY"])
        assert embedder.api_base_url == "https://api.voxell.ai/v1"
        assert embedder.model == "forge-pro"
        assert embedder.dimensions is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = ForgeTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="forge-ultra-4k",
            dimensions=512,
            prefix="START",
            suffix="END",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "https://api.voxell.ai/v1"
        assert embedder.model == "forge-ultra-4k"
        assert embedder.dimensions == 512
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "test-api-key")

        embedder_component = ForgeTextEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.forge.text_embedder.ForgeTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["FORGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "forge-pro",
                "api_base_url": "https://api.voxell.ai/v1",
                "dimensions": None,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = ForgeTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="forge-ultra-4k",
            api_base_url="https://custom-api-base-url.com",
            dimensions=512,
            prefix="START",
            suffix="END",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        component_dict = embedder.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.forge.text_embedder.ForgeTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "forge-ultra-4k",
                "api_base_url": "https://custom-api-base-url.com",
                "dimensions": 512,
                "prefix": "START",
                "suffix": "END",
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.forge.text_embedder.ForgeTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["FORGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "forge-pro",
                "api_base_url": "https://api.voxell.ai/v1",
                "dimensions": None,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        component = ForgeTextEmbedder.from_dict(data)
        assert component.api_key == Secret.from_env_var(["FORGE_API_KEY"])
        assert component.api_base_url == "https://api.voxell.ai/v1"
        assert component.model == "forge-pro"
        assert component.dimensions is None
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None

    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "test-api-key")
        embedder = ForgeTextEmbedder(model="forge-turbo", dimensions=256)
        round_tripped = ForgeTextEmbedder.from_dict(embedder.to_dict())
        assert round_tripped.to_dict() == embedder.to_dict()

    @pytest.mark.skipif(
        not os.environ.get("FORGE_API_KEY", None),
        reason="Export an env var called FORGE_API_KEY containing the Forge API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = ForgeTextEmbedder()
        text = "The food was delicious"
        result = embedder.run(text)
        assert all(isinstance(x, float) for x in result["embedding"])

    def test_run_wrong_input_format(self):
        embedder = ForgeTextEmbedder(api_key=Secret.from_token("test-api-key"))
        list_integers_input = ["text_snippet_1", "text_snippet_2"]
        match_error_msg = (
            "OpenAITextEmbedder expects a string as an input.In case you want to embed a list of Documents,"
            " please use the OpenAIDocumentEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(text=list_integers_input)
