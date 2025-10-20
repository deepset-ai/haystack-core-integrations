# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.nvidia import EmbeddingTruncateMode, NvidiaTextEmbedder
from haystack_integrations.utils.nvidia import DEFAULT_API_URL

from . import MockBackend


class TestNvidiaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        embedder = NvidiaTextEmbedder()
        embedder.warm_up()

        assert embedder.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert embedder.api_url == DEFAULT_API_URL
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = NvidiaTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="nvolveqa_40k",
            api_url="https://ai.api.nvidia.com/v1/retrieval/nvidia/test",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model == "nvolveqa_40k"
        assert embedder.api_url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/test/v1"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        embedder = NvidiaTextEmbedder("nvolveqa_40k")
        with pytest.raises(ValueError):
            embedder.warm_up()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaTextEmbedder("nvolveqa_40k")
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": DEFAULT_API_URL,
                "model": "nvolveqa_40k",
                "prefix": "",
                "suffix": "",
                "truncate": None,
                "timeout": 60.0,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaTextEmbedder(
            model="nvolveqa_40k",
            api_url="https://example.com",
            prefix="prefix",
            suffix="suffix",
            truncate=EmbeddingTruncateMode.START,
            timeout=10.0,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://example.com/v1",
                "model": "nvolveqa_40k",
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": "START",
                "timeout": 10.0,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://example.com",
                "model": "nvolveqa_40k",
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": "START",
                "timeout": 10.0,
            },
        }
        component = NvidiaTextEmbedder.from_dict(data)
        assert component.model == "nvolveqa_40k"
        assert component.api_url == "https://example.com/v1"
        assert component.prefix == "prefix"
        assert component.suffix == "suffix"
        assert component.truncate == EmbeddingTruncateMode.START
        assert component.timeout == 10.0

    def test_from_dict_defaults(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {},
        }
        component = NvidiaTextEmbedder.from_dict(data)
        ## updating this as model set to None as the warm_up is not done
        ## default model will be set post warm_up()
        assert component.model is None
        assert component.api_url == DEFAULT_API_URL
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.truncate is None

    @pytest.mark.usefixtures("mock_local_models")
    def test_run_default_model(self):
        api_key = Secret.from_token("fake-api-key")
        embedder = NvidiaTextEmbedder(api_url="http://localhost:8080/v1", api_key=api_key)

        assert embedder.model is None

        with pytest.warns(UserWarning) as record:
            embedder.warm_up()

        assert len(record) == 1
        assert "Default model is set as:" in str(record[0].message)
        assert embedder.model == "model1"

        embedder.backend = MockBackend(model=embedder.model, api_key=api_key)

        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    def test_run(self):
        api_key = Secret.from_token("fake-api-key")
        embedder = NvidiaTextEmbedder("nvidia/nv-embedqa-e5-v5", api_key=api_key, prefix="prefix ", suffix=" suffix")

        embedder.warm_up()
        embedder.backend = MockBackend(model="nvidia/nv-embedqa-e5-v5", api_key=api_key)

        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    def test_run_wrong_input_format(self):
        api_key = Secret.from_token("fake-api-key")
        embedder = NvidiaTextEmbedder("nvidia/nv-embedqa-e5-v5", api_key=api_key)
        embedder.warm_up()
        embedder.backend = MockBackend(model="nvidia/nv-embedqa-e5-v5", api_key=api_key)

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="NvidiaTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    def test_run_empty_string(self):
        model = "nvidia/nv-embedqa-e5-v5"
        api_key = Secret.from_token("fake-api-key")
        embedder = NvidiaTextEmbedder(model, api_key=api_key)

        embedder.warm_up()
        embedder.backend = MockBackend(model=model, api_key=api_key)

        with pytest.raises(ValueError, match="empty string"):
            embedder.run(text="")

    def test_setting_timeout(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        embedder = NvidiaTextEmbedder(timeout=10.0)
        embedder.warm_up()
        assert embedder.backend.timeout == 10.0

    def test_setting_timeout_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        monkeypatch.setenv("NVIDIA_TIMEOUT", "45")
        embedder = NvidiaTextEmbedder()
        embedder.warm_up()
        assert embedder.backend.timeout == 45.0

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_NIM_EMBEDDER_MODEL", None) or not os.environ.get("NVIDIA_NIM_ENDPOINT_URL", None),
        reason="Export an env var called NVIDIA_NIM_EMBEDDER_MODEL containing the hosted model name and "
        "NVIDIA_NIM_ENDPOINT_URL containing the local URL to call.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nim_backend(self):
        model = os.environ["NVIDIA_NIM_EMBEDDER_MODEL"]
        url = os.environ["NVIDIA_NIM_ENDPOINT_URL"]
        embedder = NvidiaTextEmbedder(
            model=model,
            api_url=url,
            api_key=None,
        )
        embedder.warm_up()

        result = embedder.run("A transformer is a deep learning architecture")
        embedding = result["embedding"]
        meta = result["meta"]

        assert all(isinstance(x, float) for x in embedding)
        assert "usage" in meta

    @pytest.mark.parametrize(
        "model, api_url",
        [
            ("nvidia/nv-embedqa-e5-v5", "https://integrate.api.nvidia.com/v1"),
        ],
        ids=[
            "nvidia/nv-embedqa-e5-v5",
        ],
    )
    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_api_catalog(self, model, api_url):
        embedder = NvidiaTextEmbedder(
            model=model,
            **({"api_url": api_url} if api_url else {}),
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
        )
        embedder.warm_up()

        result = embedder.run("A transformer is a deep learning architecture")
        embedding = result["embedding"]
        meta = result["meta"]

        assert all(isinstance(x, float) for x in embedding)
        assert "usage" in meta
