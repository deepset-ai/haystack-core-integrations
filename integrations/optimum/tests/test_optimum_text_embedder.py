from unittest.mock import MagicMock, patch

import pytest
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders import OptimumTextEmbedder
from huggingface_hub.utils import RepositoryNotFoundError


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack_integrations.components.embedders.optimum_text_embedder.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


class TestOptimumTextEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model):  # noqa: ARG002
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = OptimumTextEmbedder()

        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.normalize_embeddings is True
        assert embedder.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder.model_kwargs == {
            "model_id": "BAAI/bge-small-en-v1.5",
            "provider": "CPUExecutionProvider",
            "use_auth_token": "fake-api-token",
        }

    def test_init_with_parameters(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumTextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            normalize_embeddings=False,
            onnx_execution_provider="CUDAExecutionProvider",
            model_kwargs={"trust_remote_code": True},
        )

        assert embedder.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.normalize_embeddings is False
        assert embedder.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder.model_kwargs == {
            "trust_remote_code": True,
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CUDAExecutionProvider",
            "use_auth_token": "fake-api-token",
        }

    def test_initialize_with_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            OptimumTextEmbedder(model="invalid_model_id")

    def test_to_dict(self, mock_check_valid_model):  # noqa: ARG002
        component = OptimumTextEmbedder()
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum_text_embedder.OptimumTextEmbedder",
            "init_parameters": {
                "model": "BAAI/bge-small-en-v1.5",
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "",
                "suffix": "",
                "normalize_embeddings": True,
                "onnx_execution_provider": "CPUExecutionProvider",
                "model_kwargs": {
                    "model_id": "BAAI/bge-small-en-v1.5",
                    "provider": "CPUExecutionProvider",
                    "use_auth_token": None,
                },
            },
        }

    def test_to_dict_with_custom_init_parameters(self, mock_check_valid_model):  # noqa: ARG002
        component = OptimumTextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            normalize_embeddings=False,
            onnx_execution_provider="CUDAExecutionProvider",
            model_kwargs={"trust_remote_code": True},
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum_text_embedder.OptimumTextEmbedder",
            "init_parameters": {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "normalize_embeddings": False,
                "onnx_execution_provider": "CUDAExecutionProvider",
                "model_kwargs": {
                    "trust_remote_code": True,
                    "model_id": "sentence-transformers/all-mpnet-base-v2",
                    "provider": "CUDAExecutionProvider",
                    "use_auth_token": None,
                },
            },
        }

    def test_run_wrong_input_format(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumTextEmbedder(
            model="BAAI/bge-small-en-v1.5",
            token=Secret.from_token("fake-api-token"),
        )
        embedder.warm_up()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OptimumTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run(self):
        embedder = OptimumTextEmbedder(
            model="BAAI/bge-small-en-v1.5",
            prefix="prefix ",
            suffix=" suffix",
        )
        embedder.warm_up()

        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])
