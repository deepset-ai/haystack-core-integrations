import os

import pytest
from haystack.utils import Secret
from haystack_integrations.components.embedders.nvidia import NvidiaEmbeddingModel, NvidiaTextEmbedder
from haystack_integrations.utils.nvidia.client import AvailableNvidiaCloudFunctions


class MockClient:
    def query_function(self, func_id, payload):
        data = [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
        return {"data": data, "usage": {"total_tokens": 4, "prompt_tokens": 4}}

    def available_functions(self):
        return {
            NvidiaEmbeddingModel.NVOLVE_40K.value: AvailableNvidiaCloudFunctions(
                name=NvidiaEmbeddingModel.NVOLVE_40K.value, id="fake-id", status="ACTIVE"
            )
        }


class TestNvidiaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        embedder = NvidiaTextEmbedder(NvidiaEmbeddingModel.NVOLVE_40K)

        assert embedder.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert embedder.model == NvidiaEmbeddingModel.NVOLVE_40K
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = NvidiaTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="playground_nvolveqa_40k",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model == NvidiaEmbeddingModel.NVOLVE_40K
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            NvidiaTextEmbedder(NvidiaEmbeddingModel.NVOLVE_40K)

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaTextEmbedder(NvidiaEmbeddingModel.NVOLVE_40K)
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "playground_nvolveqa_40k",
                "prefix": "",
                "suffix": "",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaTextEmbedder(
            model=NvidiaEmbeddingModel.NVOLVE_40K,
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "playground_nvolveqa_40k",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    def test_run(self):
        embedder = NvidiaTextEmbedder(
            "playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"), prefix="prefix ", suffix=" suffix"
        )
        embedder.client = MockClient()
        embedder.warm_up()
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    def test_run_wrong_input_format(self):
        embedder = NvidiaTextEmbedder("playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"))
        embedder.client = MockClient()
        embedder.warm_up()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="NvidiaTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        embedder = NvidiaTextEmbedder("playground_nvolveqa_40k")
        embedder.warm_up()

        result = embedder.run("A transformer is a deep learning architecture")
        embedding = result["embedding"]
        meta = result["meta"]

        assert all(isinstance(x, float) for x in embedding)
        assert "usage" in meta
