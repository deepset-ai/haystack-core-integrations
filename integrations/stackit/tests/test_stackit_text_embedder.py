# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.stackit.text_embedder import STACKITTextEmbedder


class TestSTACKITTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")

        embedder = STACKITTextEmbedder(model="intfloat/e5-mistral-7b-instruct")
        assert embedder.api_key == Secret.from_env_var(["STACKIT_API_KEY"])
        assert embedder.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert embedder.model == "intfloat/e5-mistral-7b-instruct"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = STACKITTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="intfloat/e5-mistral-7b-instruct",
            prefix="START",
            suffix="END",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert embedder.model == "intfloat/e5-mistral-7b-instruct"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")

        embedder_component = STACKITTextEmbedder(model="intfloat/e5-mistral-7b-instruct")
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "intfloat/e5-mistral-7b-instruct",
                "api_base_url": "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = STACKITTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="intfloat/e5-mistral-7b-instruct",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
        )
        component_dict = embedder.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "intfloat/e5-mistral-7b-instruct",
                "api_base_url": "https://custom-api-base-url.com",
                "prefix": "START",
                "suffix": "END",
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "https://proxy.example.com"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-secret-key")
        data = {
            "type": "haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "intfloat/e5-mistral-7b-instruct",
                "api_base_url": "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        embedder = STACKITTextEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var(["STACKIT_API_KEY"])
        assert embedder.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert embedder.model == "intfloat/e5-mistral-7b-instruct"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = STACKITTextEmbedder(model="intfloat/e5-mistral-7b-instruct")
        text = "The food was delicious"
        result = embedder.run(text)
        assert all(isinstance(x, float) for x in result["embedding"])

    def test_run_wrong_input_format(self):
        embedder = STACKITTextEmbedder(
            model="intfloat/e5-mistral-7b-instruct", api_key=Secret.from_token("test-api-key")
        )
        list_integers_input = ["text_snippet_1", "text_snippet_2"]
        match_error_msg = (
            "OpenAITextEmbedder expects a string as an input.In case you want to embed a list of Documents,"
            " please use the OpenAIDocumentEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(text=list_integers_input)
