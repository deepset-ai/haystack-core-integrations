import os

import pytest
from haystack.utils import Secret
from haystack_integrations.components.embedders.nvidia import EmbeddingTruncateMode, NvidiaTextEmbedder
from haystack_integrations.components.embedders.nvidia.backend import EmbedderBackend


class MockBackend(EmbedderBackend):
    def __init__(self, model, model_kwargs):
        super().__init__(model, model_kwargs)

    def embed(self, texts):
        inputs = texts
        data = [[0.1, 0.2, 0.3] for i in range(len(inputs))]
        return data, {"usage": {"total_tokens": 4, "prompt_tokens": 4}}


class TestNvidiaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        embedder = NvidiaTextEmbedder()

        assert embedder.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert embedder.model == "NV-Embed-QA"
        assert embedder.api_url == "https://ai.api.nvidia.com/v1/retrieval/nvidia"
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
        assert embedder.api_url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/test"
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
                "api_url": "https://ai.api.nvidia.com/v1/retrieval/nvidia",
                "model": "nvolveqa_40k",
                "prefix": "",
                "suffix": "",
                "truncate": None,
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
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://example.com",
                "model": "nvolveqa_40k",
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": "START",
            },
        }

    def from_dict(self, monkeypatch):
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
            },
        }
        component = NvidiaTextEmbedder.from_dict(data)
        assert component.model == "nvolveqa_40k"
        assert component.api_url == "https://example.com"
        assert component.prefix == "prefix"
        assert component.suffix == "suffix"
        assert component.truncate == "START"

    def test_run(self):
        embedder = NvidiaTextEmbedder(
            "playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"), prefix="prefix ", suffix=" suffix"
        )

        embedder.warm_up()
        embedder.backend = MockBackend("aa", None)

        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    def test_run_wrong_input_format(self):
        embedder = NvidiaTextEmbedder("playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"))
        embedder.warm_up()
        embedder.backend = MockBackend("aa", None)

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="NvidiaTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

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

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_api_catalog(self):
        embedder = NvidiaTextEmbedder(
            model="NV-Embed-QA",
            api_url="https://ai.api.nvidia.com/v1/retrieval/nvidia",
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
        )
        embedder.warm_up()

        result = embedder.run("A transformer is a deep learning architecture")
        embedding = result["embedding"]
        meta = result["meta"]

        assert all(isinstance(x, float) for x in embedding)
        assert "usage" in meta
