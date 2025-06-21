from unittest.mock import MagicMock, patch

import pytest
from haystack.utils.auth import Secret
from huggingface_hub.utils import RepositoryNotFoundError

from haystack_integrations.components.embedders.optimum import OptimumTextEmbedder
from haystack_integrations.components.embedders.optimum.optimization import (
    OptimumEmbedderOptimizationConfig,
    OptimumEmbedderOptimizationMode,
)
from haystack_integrations.components.embedders.optimum.pooling import OptimumEmbedderPooling
from haystack_integrations.components.embedders.optimum.quantization import (
    OptimumEmbedderQuantizationConfig,
    OptimumEmbedderQuantizationMode,
)


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack_integrations.components.embedders.optimum._backend.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


@pytest.fixture
def mock_get_pooling_mode():
    with patch(
        "haystack_integrations.components.embedders.optimum._backend._pooling_from_model_config",
        MagicMock(return_value=OptimumEmbedderPooling.MEAN),
    ) as mock:
        yield mock


class TestOptimumTextEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model, mock_get_pooling_mode):  # noqa: ARG002
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = OptimumTextEmbedder()

        assert embedder._backend.parameters.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder._backend.parameters.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder._backend.parameters.prefix == ""
        assert embedder._backend.parameters.suffix == ""
        assert embedder._backend.parameters.normalize_embeddings is True
        assert embedder._backend.parameters.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN
        assert embedder._backend.parameters.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
            "token": "fake-api-token",
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
            working_dir="working_dir",
            optimizer_settings=None,
            quantizer_settings=None,
        )

        assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._backend.parameters.token == Secret.from_token("fake-api-token")
        assert embedder._backend.parameters.prefix == "prefix"
        assert embedder._backend.parameters.suffix == "suffix"
        assert embedder._backend.parameters.normalize_embeddings is False
        assert embedder._backend.parameters.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MAX
        assert embedder._backend.parameters.model_kwargs == {
            "trust_remote_code": True,
            "model_id": "sentence-transformers/all-minilm-l6-v2",
            "provider": "CUDAExecutionProvider",
            "token": "fake-api-token",
        }
        assert embedder._backend.parameters.working_dir == "working_dir"
        assert embedder._backend.parameters.optimizer_settings is None
        assert embedder._backend.parameters.quantizer_settings is None

    def test_to_and_from_dict(self, mock_check_valid_model, mock_get_pooling_mode, monkeypatch):  # noqa: ARG002
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
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
                "working_dir": None,
                "model_kwargs": {
                    "model_id": "sentence-transformers/all-mpnet-base-v2",
                    "provider": "CPUExecutionProvider",
                },
                "optimizer_settings": None,
                "quantizer_settings": None,
            },
        }

        embedder = OptimumTextEmbedder.from_dict(data)
        assert embedder._backend.parameters.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder._backend.parameters.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder._backend.parameters.prefix == ""
        assert embedder._backend.parameters.suffix == ""
        assert embedder._backend.parameters.normalize_embeddings is True
        assert embedder._backend.parameters.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN
        assert embedder._backend.parameters.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
            "token": None,
        }
        assert embedder._backend.parameters.working_dir is None
        assert embedder._backend.parameters.optimizer_settings is None
        assert embedder._backend.parameters.quantizer_settings is None

    def test_to_and_from_dict_with_custom_init_parameters(self, mock_check_valid_model):  # noqa: ARG002
        component = OptimumTextEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            normalize_embeddings=False,
            onnx_execution_provider="CUDAExecutionProvider",
            pooling_mode="max",
            model_kwargs={"trust_remote_code": True},
            working_dir="working_dir",
            optimizer_settings=OptimumEmbedderOptimizationConfig(OptimumEmbedderOptimizationMode.O1, for_gpu=True),
            quantizer_settings=OptimumEmbedderQuantizationConfig(
                OptimumEmbedderQuantizationMode.ARM64, per_channel=True
            ),
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
                },
                "working_dir": "working_dir",
                "optimizer_settings": {"mode": "o1", "for_gpu": True},
                "quantizer_settings": {"mode": "arm64", "per_channel": True},
            },
        }

        embedder = OptimumTextEmbedder.from_dict(data)
        assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._backend.parameters.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert embedder._backend.parameters.prefix == "prefix"
        assert embedder._backend.parameters.suffix == "suffix"
        assert embedder._backend.parameters.normalize_embeddings is False
        assert embedder._backend.parameters.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MAX
        assert embedder._backend.parameters.model_kwargs == {
            "trust_remote_code": True,
            "model_id": "sentence-transformers/all-minilm-l6-v2",
            "provider": "CUDAExecutionProvider",
            "token": None,
        }
        assert embedder._backend.parameters.working_dir == "working_dir"
        assert embedder._backend.parameters.optimizer_settings == OptimumEmbedderOptimizationConfig(
            OptimumEmbedderOptimizationMode.O1, for_gpu=True
        )
        assert embedder._backend.parameters.quantizer_settings == OptimumEmbedderQuantizationConfig(
            OptimumEmbedderQuantizationMode.ARM64, per_channel=True
        )

    def test_initialize_with_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            OptimumTextEmbedder(model="invalid_model_id", pooling_mode="max")

    def test_initialize_with_invalid_pooling_mode(self, mock_check_valid_model):  # noqa: ARG002
        mock_get_pooling_mode.side_effect = ValueError("Invalid pooling mode")
        with pytest.raises(ValueError):
            OptimumTextEmbedder(model="sentence-transformers/all-mpnet-base-v2", pooling_mode="Invalid_pooling_mode")

    def test_infer_pooling_mode_from_str(self, mock_check_valid_model):  # noqa: ARG002
        """
        Test that the pooling mode is correctly inferred from a string.
        The pooling mode is "mean" as per the model config.
        """
        for pooling_mode in OptimumEmbedderPooling:
            embedder = OptimumTextEmbedder(
                model="sentence-transformers/all-minilm-l6-v2",
                pooling_mode=pooling_mode.value,
            )

            assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
            assert embedder._backend.parameters.pooling_mode == pooling_mode

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

        assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN

    def test_run_wrong_input_format(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumTextEmbedder(
            model="sentence-transformers/paraphrase-albert-small-v2",
            token=Secret.from_token("fake-api-token"),
            pooling_mode="mean",
        )
        embedder._initialized = True

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OptimumTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run(self):
        for pooling_mode in OptimumEmbedderPooling:
            embedder = OptimumTextEmbedder(
                model="sentence-transformers/paraphrase-albert-small-v2",
                prefix="prefix ",
                suffix=" suffix",
                pooling_mode=pooling_mode,
            )
            embedder.warm_up()

            result = embedder.run(text="The food was delicious")

            assert len(result["embedding"]) == 768
            assert all(isinstance(x, float) for x in result["embedding"])
