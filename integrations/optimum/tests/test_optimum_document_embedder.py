# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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


class TestInitializationSerialization:
    def test_init_default(self):
        embedder = OptimumDocumentEmbedder()

        assert embedder._params.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder._params.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder._params.prefix == ""
        assert embedder._params.suffix == ""
        assert embedder._params.normalize_embeddings is True
        assert embedder._params.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder._params.pooling_mode is None
        assert embedder._params.batch_size == 32
        assert embedder._params.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder._params.model_kwargs is None
        assert embedder._backend is None

    def test_init_with_parameters(self):
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

        assert embedder._params.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._params.token == Secret.from_token("fake-api-token")
        assert embedder._params.prefix == "prefix"
        assert embedder._params.suffix == "suffix"
        assert embedder._params.batch_size == 64
        assert embedder._params.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder._params.normalize_embeddings is False
        assert embedder._params.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder._params.pooling_mode == "max"
        assert embedder._params.model_kwargs == {"trust_remote_code": True}
        assert embedder._params.working_dir == "working_dir"
        assert embedder._params.optimizer_settings is None
        assert embedder._params.quantizer_settings is None
        assert embedder._backend is None

    def test_to_and_from_dict(self):
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
                "pooling_mode": None,
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
        assert embedder._params.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder._params.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder._params.prefix == ""
        assert embedder._params.suffix == ""
        assert embedder._params.normalize_embeddings is True
        assert embedder._params.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder._params.pooling_mode is None
        assert embedder._params.batch_size == 32
        assert embedder._params.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder._params.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
        }
        assert embedder._params.working_dir is None
        assert embedder._params.optimizer_settings is None
        assert embedder._params.quantizer_settings is None
        assert embedder._backend is None

    def test_to_and_from_dict_with_custom_init_parameters(self):
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
        assert embedder._params.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._params.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert embedder._params.prefix == "prefix"
        assert embedder._params.suffix == "suffix"
        assert embedder._params.batch_size == 64
        assert embedder._params.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder._params.normalize_embeddings is False
        assert embedder._params.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder._params.pooling_mode == OptimumEmbedderPooling.MAX
        assert embedder._params.model_kwargs == {
            "trust_remote_code": True,
            "model_id": "sentence-transformers/all-minilm-l6-v2",
            "provider": "CUDAExecutionProvider",
        }
        assert embedder._params.working_dir == "working_dir"
        assert embedder._params.optimizer_settings == OptimumEmbedderOptimizationConfig(
            OptimumEmbedderOptimizationMode.O1, for_gpu=True
        )
        assert embedder._params.quantizer_settings == OptimumEmbedderQuantizationConfig(
            OptimumEmbedderQuantizationMode.ARM64, per_channel=True
        )
        assert embedder._backend is None

    def test_from_dict_with_legacy_serialization(self):
        data = {
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

        assert embedder._params.pooling_mode == OptimumEmbedderPooling.MEAN
        assert embedder._params.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
        }
        assert embedder._params.batch_size == 32
        assert embedder._backend is None


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        embedder = OptimumDocumentEmbedder(
            token=Secret.from_env_var("HF_API_TOKEN"),
            pooling_mode="mean",
        )

        assert embedder._backend is None
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
            embedder.warm_up()
        assert embedder._backend is None

    def test_warm_up_is_idempotent(self):
        backend = MagicMock()
        with patch(
            "haystack_integrations.components.embedders.optimum.optimum_document_embedder._EmbedderBackend",
            return_value=backend,
        ) as backend_class:
            embedder = OptimumDocumentEmbedder(pooling_mode="mean")
            embedder.warm_up()
            embedder.warm_up()

        backend_class.assert_called_once()
        backend.warm_up.assert_called_once()
        assert embedder._backend is backend

    def test_warm_up_with_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        embedder = OptimumDocumentEmbedder(model="invalid_model_id")
        with pytest.raises(RepositoryNotFoundError):
            embedder.warm_up()

    def test_warm_up_with_invalid_pooling_mode(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2", pooling_mode="Invalid_pooling_mode"
        )
        with pytest.raises(ValueError):
            embedder.warm_up()


class TestRun:

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
            with patch("haystack_integrations.components.embedders.optimum._backend._EmbedderBackend.warm_up"):
                embedder.warm_up()

            assert embedder._backend is not None
            assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
            assert embedder._backend.parameters.pooling_mode == pooling_mode

    @pytest.mark.integration
    def test_default_pooling_mode_when_config_not_found(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumDocumentEmbedder(model="embedding_model_finetuned", pooling_mode=None)
        with pytest.raises(ValueError):
            embedder.warm_up()

    @pytest.mark.integration
    def test_infer_pooling_mode_from_hf(self):
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            pooling_mode=None,
        )
        embedder.warm_up()

        assert embedder._backend is not None
        assert embedder._backend.parameters.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder._backend.parameters.pooling_mode == OptimumEmbedderPooling.MEAN

    def test_prepare_texts_to_embed_w_metadata(self):
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

    def test_prepare_texts_to_embed_w_suffix(self):
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

    def test_run_wrong_input_format(self):
        embedder = OptimumDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2", pooling_mode="mean")
        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OptimumDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OptimumDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self):
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/paraphrase-albert-small-v2",
        )
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
