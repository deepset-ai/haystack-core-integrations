import copy
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from huggingface_hub.utils import RepositoryNotFoundError

from haystack_integrations.components.embedders.optimum import OptimumDocumentEmbedder
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


class TestOptimumDocumentEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model, mock_get_pooling_mode):  # noqa: ARG002
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = OptimumDocumentEmbedder()

        assert embedder._backend.parameters.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder._backend.parameters.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder._backend.parameters.prefix == ""
        assert embedder._backend.parameters.suffix == ""
        assert embedder._backend.parameters.normalize_embeddings is True
        assert embedder._backend.parameters.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN
        assert embedder._backend.parameters.batch_size == 32
        assert embedder._backend.parameters.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder._backend.parameters.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
            "token": "fake-api-token",
        }

    def test_init_with_parameters(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
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
        assert embedder._backend.parameters.batch_size == 64
        assert embedder._backend.parameters.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
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
        component = OptimumDocumentEmbedder()
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder",
            "init_parameters": {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "normalize_embeddings": True,
                "onnx_execution_provider": "CPUExecutionProvider",
                "pooling_mode": "mean",
                "model_kwargs": {
                    "model_id": "sentence-transformers/all-mpnet-base-v2",
                    "provider": "CPUExecutionProvider",
                },
                "working_dir": None,
                "optimizer_settings": None,
                "quantizer_settings": None,
            },
        }

        embedder = OptimumDocumentEmbedder.from_dict(data)
        assert embedder._backend.parameters.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder._backend.parameters.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder._backend.parameters.prefix == ""
        assert embedder._backend.parameters.suffix == ""
        assert embedder._backend.parameters.normalize_embeddings is True
        assert embedder._backend.parameters.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN
        assert embedder._backend.parameters.batch_size == 32
        assert embedder._backend.parameters.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder._backend.parameters.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
            "token": None,
        }
        assert embedder._backend.parameters.working_dir is None
        assert embedder._backend.parameters.optimizer_settings is None
        assert embedder._backend.parameters.quantizer_settings is None

    def test_to_and_from_dict_with_custom_init_parameters(self, mock_check_valid_model, mock_get_pooling_mode):
        component = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
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
            "type": "haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder",
            "init_parameters": {
                "model": "sentence-transformers/all-minilm-l6-v2",
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
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

        embedder = OptimumDocumentEmbedder.from_dict(data)
        assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._backend.parameters.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert embedder._backend.parameters.prefix == "prefix"
        assert embedder._backend.parameters.suffix == "suffix"
        assert embedder._backend.parameters.batch_size == 64
        assert embedder._backend.parameters.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
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
            OptimumDocumentEmbedder(model="invalid_model_id")

    def test_initialize_with_invalid_pooling_mode(self, mock_check_valid_model):  # noqa: ARG002
        mock_get_pooling_mode.side_effect = ValueError("Invalid pooling mode")
        with pytest.raises(ValueError):
            OptimumDocumentEmbedder(
                model="sentence-transformers/all-mpnet-base-v2", pooling_mode="Invalid_pooling_mode"
            )

    def test_infer_pooling_mode_from_str(self, mock_check_valid_model):  # noqa: ARG002
        """
        Test that the pooling mode is correctly inferred from a string.
        The pooling mode is "mean" as per the model config.
        """
        for pooling_mode in OptimumEmbedderPooling:
            embedder = OptimumDocumentEmbedder(
                model="sentence-transformers/all-minilm-l6-v2",
                pooling_mode=pooling_mode.value,
            )

            assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
            assert embedder._backend.parameters.pooling_mode == pooling_mode

    @pytest.mark.integration
    def test_default_pooling_mode_when_config_not_found(self, mock_check_valid_model):  # noqa: ARG002
        with pytest.raises(ValueError):
            OptimumDocumentEmbedder(
                model="embedding_model_finetuned",
                pooling_mode=None,
            )

    @pytest.mark.integration
    def test_infer_pooling_mode_from_hf(self):
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            pooling_mode=None,
        )

        assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN

    def test_prepare_texts_to_embed_w_metadata(self, mock_check_valid_model):  # noqa: ARG002
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
            pooling_mode="mean",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    def test_prepare_texts_to_embed_w_suffix(self, mock_check_valid_model):  # noqa: ARG002
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            prefix="my_prefix ",
            suffix=" my_suffix",
            pooling_mode="mean",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_run_wrong_input_format(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2", pooling_mode="mean")
        embedder._initialized = True
        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OptimumDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OptimumDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/paraphrase-albert-small-v2",
        )
        embedder._initialized = True
        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "opt_config, quant_config",
        [
            (None, None),
            (
                OptimumEmbedderOptimizationConfig(OptimumEmbedderOptimizationMode.O1, for_gpu=False),
                None,
            ),
            (None, OptimumEmbedderQuantizationConfig(OptimumEmbedderQuantizationMode.AVX2)),
            # onxxruntime 1.17.x breaks support for quantizing optimized models.
            # c.f https://discuss.huggingface.co/t/optimize-and-quantize-with-optimum/23675/12
            # (
            #     OptimumEmbedderOptimizationConfig(OptimumEmbedderOptimizationMode.O2, for_gpu=False),
            #     OptimumEmbedderQuantizationConfig(OptimumEmbedderQuantizationMode.AVX2),
            # ),
        ],
    )
    def test_run(self, opt_config, quant_config):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
            Document(content="Every planet we reach is dead", meta={"topic": "Monkeys"}),
        ]
        docs_copy = copy.deepcopy(docs)

        with tempfile.TemporaryDirectory() as tmpdirname:
            embedder = OptimumDocumentEmbedder(
                model="sentence-transformers/paraphrase-albert-small-v2",
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
                working_dir=tmpdirname,
                optimizer_settings=opt_config,
                quantizer_settings=quant_config,
            )
            embedder.warm_up()

            result = embedder.run(documents=docs)
            _ = [embedder.run([d]) for d in docs_copy]

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 768
            assert all(isinstance(x, float) for x in doc.embedding)

        # Check order
        assert [d.embedding for d in docs_copy] == [d.embedding for d in docs]
