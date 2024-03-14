import os
from unittest.mock import Mock, patch

import pytest
from haystack.utils import Secret
from haystack_integrations.components.embedders.nvidia import NvidiaTextEmbedder


class TestNvidiaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        embedder = NvidiaTextEmbedder("nvolveqa_40k")

        assert embedder.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert embedder.model == "nvolveqa_40k"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = NvidiaTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="nvolveqa_40k",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model == "nvolveqa_40k"
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
                "api_url": None,
                "model": "nvolveqa_40k",
                "prefix": "",
                "suffix": "",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaTextEmbedder(
            model="nvolveqa_40k",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": None,
                "model": "nvolveqa_40k",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run(self, mock_client_class):
        embedder = NvidiaTextEmbedder(
            "playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"), prefix="prefix ", suffix=" suffix"
        )
        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=Mock(
                return_value={
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"total_tokens": 4, "prompt_tokens": 4},
                }
            ),
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run_wrong_input_format(self, mock_client_class):
        embedder = NvidiaTextEmbedder("playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"))
        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=Mock(
                return_value={
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"total_tokens": 4, "prompt_tokens": 4},
                }
            ),
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="NvidiaTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nvcf_backend(self):
        embedder = NvidiaTextEmbedder("playground_nvolveqa_40k")
        embedder.warm_up()

        result = embedder.run("A transformer is a deep learning architecture")
        embedding = result["embedding"]
        meta = result["meta"]

        assert all(isinstance(x, float) for x in embedding)
        assert "usage" in meta

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nim_backend(self):
        embedder = NvidiaTextEmbedder(
            model="text-embedding-ada-002",
            api_url="https://api.openai.com/v1",
            api_key=Secret.from_env_var(["OPENAI_API_KEY"]),
        )
        embedder.warm_up()

        result = embedder.run("A transformer is a deep learning architecture")
        embedding = result["embedding"]
        meta = result["meta"]

        assert all(isinstance(x, float) for x in embedding)
        assert "usage" in meta
