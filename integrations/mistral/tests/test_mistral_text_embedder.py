# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils import Secret
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder

pytestmark = pytest.mark.embedders


class TestMistralTextEmbedder:
    def test_init_default(self):
        embedder = MistralTextEmbedder()
        assert embedder.api_key == Secret.from_env_var(["MISTRAL_API_KEY"])
        assert embedder.model == "mistral-embed"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = MistralTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-embed-v2",
            prefix="START",
            suffix="END",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "mistral-embed-v2"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"

    def test_to_dict(self):
        embedder_component = MistralTextEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-embed",
                "dimensions": None,
                "organization": None,
                "prefix": "",
                "suffix": "",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = MistralTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="mistral-embed-v2",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
        )
        component_dict = embedder.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "mistral-embed-v2",
                "dimensions": None,
                "organization": None,
                "prefix": "START",
                "suffix": "END",
            },
        }

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = MistralTextEmbedder()
        text = "The food was delicious"
        result = embedder.run(text)
        assert all(isinstance(x, float) for x in result["embedding"])

    def test_run_wrong_input_format(self):
        embedder = MistralTextEmbedder(api_key=Secret.from_token("test-api-key"))
        list_integers_input = ["text_snippet_1", "text_snippet_2"]
        match_error_msg = (
            "OpenAITextEmbedder expects a string as an input.In case you want to embed a list of Documents,"
            " please use the OpenAIDocumentEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(text=list_integers_input)
