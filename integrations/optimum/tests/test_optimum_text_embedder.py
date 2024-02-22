from unittest.mock import MagicMock, patch

import pytest
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.optimum import OptimumTextEmbedder
from haystack_integrations.components.embedders.optimum.pooling import PoolingMode
from huggingface_hub.utils import RepositoryNotFoundError


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack_integrations.components.embedders.optimum.optimum_text_embedder.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


@pytest.fixture
def mock_get_pooling_mode():
    with patch(
        "haystack_integrations.components.embedders.optimum.optimum_text_embedder.HFPoolingMode.get_pooling_mode",
        MagicMock(return_value=PoolingMode.MEAN),
    ) as mock:
        yield mock


class TestOptimumTextEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model, mock_get_pooling_mode):  # noqa: ARG002
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = OptimumTextEmbedder()

        assert embedder.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.normalize_embeddings is True
        assert embedder.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder.pooling_mode == PoolingMode.MEAN
        assert embedder.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
            "use_auth_token": "fake-api-token",
        }

    def test_init_with_parameters(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumTextEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            normalize_embeddings=False,
            pooling_mode="max",
            onnx_execution_provider="CUDAExecutionProvider",
            model_kwargs={"trust_remote_code": True},
        )

        assert embedder.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.normalize_embeddings is False
        assert embedder.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder.pooling_mode == PoolingMode.MAX
        assert embedder.model_kwargs == {
            "trust_remote_code": True,
            "model_id": "sentence-transformers/all-minilm-l6-v2",
            "provider": "CUDAExecutionProvider",
            "use_auth_token": "fake-api-token",
        }

    def test_to_dict(self, mock_check_valid_model, mock_get_pooling_mode):  # noqa: ARG002
        component = OptimumTextEmbedder()
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder",
            "init_parameters": {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "",
                "suffix": "",
                "normalize_embeddings": True,
                "onnx_execution_provider": "CPUExecutionProvider",
                "pooling_mode": "mean",
                "model_kwargs": {
                    "model_id": "sentence-transformers/all-mpnet-base-v2",
                    "provider": "CPUExecutionProvider",
                    "use_auth_token": None,
                },
            },
        }

    def test_to_dict_with_custom_init_parameters(self, mock_check_valid_model):  # noqa: ARG002
        component = OptimumTextEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            normalize_embeddings=False,
            onnx_execution_provider="CUDAExecutionProvider",
            pooling_mode="max",
            model_kwargs={"trust_remote_code": True},
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder",
            "init_parameters": {
                "model": "sentence-transformers/all-minilm-l6-v2",
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "normalize_embeddings": False,
                "onnx_execution_provider": "CUDAExecutionProvider",
                "pooling_mode": "max",
                "model_kwargs": {
                    "trust_remote_code": True,
                    "model_id": "sentence-transformers/all-minilm-l6-v2",
                    "provider": "CUDAExecutionProvider",
                    "use_auth_token": None,
                },
            },
        }

    def test_initialize_with_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            OptimumTextEmbedder(model="invalid_model_id", pooling_mode="max")

    def test_initialize_with_invalid_pooling_mode(self, mock_check_valid_model):  # noqa: ARG002
        mock_get_pooling_mode.side_effect = ValueError("Invalid pooling mode")
        with pytest.raises(ValueError):
            OptimumTextEmbedder(model="sentence-transformers/all-mpnet-base-v2", pooling_mode="Invalid_pooling_mode")

    def test_infer_pooling_mode_from_str(self):
        """
        Test that the pooling mode is correctly inferred from a string.
        The pooling mode is "mean" as per the model config.
        """
        for pooling_mode in PoolingMode:
            embedder = OptimumTextEmbedder(
                model="sentence-transformers/all-minilm-l6-v2",
                pooling_mode=pooling_mode.value,
            )

            assert embedder.model == "sentence-transformers/all-minilm-l6-v2"
            assert embedder.pooling_mode == pooling_mode

    @pytest.mark.integration
    def test_default_pooling_mode_when_config_not_found(self, mock_check_valid_model):  # noqa: ARG002
        with pytest.raises(ValueError):
            OptimumTextEmbedder(
                model="embedding_model_finetuned",
                pooling_mode=None,
            )

    @pytest.mark.integration
    def test_infer_pooling_mode_from_hf(self):
        embedder = OptimumTextEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            pooling_mode=None,
        )

        assert embedder.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder.pooling_mode == PoolingMode.MEAN

    def test_run_wrong_input_format(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumTextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            token=Secret.from_token("fake-api-token"),
            pooling_mode="mean",
        )
        embedder.warm_up()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OptimumTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run(self):
        embedder = OptimumTextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            prefix="prefix ",
            suffix=" suffix",
        )
        embedder.warm_up()

        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 768
        assert all(isinstance(x, float) for x in result["embedding"])
