from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders import OptimumDocumentEmbedder
from huggingface_hub.utils import RepositoryNotFoundError


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack_integrations.components.embedders.optimum_document_embedder.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


class TestOptimumDocumentEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model):  # noqa: ARG002
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = OptimumDocumentEmbedder()

        assert embedder.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.normalize_embeddings is True
        assert embedder.onnx_execution_provider == "CPUExecutionProvider"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.model_kwargs == {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "provider": "CPUExecutionProvider",
            "use_auth_token": "fake-api-token",
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
            onnx_execution_provider="CUDAExecutionProvider",
            model_kwargs={"trust_remote_code": True},
        )

        assert embedder.model == "sentence-transformers/all-minilm-l6-v2"
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder.normalize_embeddings is False
        assert embedder.onnx_execution_provider == "CUDAExecutionProvider"
        assert embedder.model_kwargs == {
            "trust_remote_code": True,
            "model_id": "sentence-transformers/all-minilm-l6-v2",
            "provider": "CUDAExecutionProvider",
            "use_auth_token": "fake-api-token",
        }

    def test_initialize_with_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            OptimumDocumentEmbedder(model="invalid_model_id")

    def test_to_dict(self, mock_check_valid_model):  # noqa: ARG002
        component = OptimumDocumentEmbedder()
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum_document_embedder.OptimumDocumentEmbedder",
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
                "model_kwargs": {
                    "model_id": "sentence-transformers/all-mpnet-base-v2",
                    "provider": "CPUExecutionProvider",
                    "use_auth_token": None,
                },
            },
        }

    def test_to_dict_with_custom_init_parameters(self, mock_check_valid_model):  # noqa: ARG002
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
            model_kwargs={"trust_remote_code": True},
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.optimum_document_embedder.OptimumDocumentEmbedder",
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
                "model_kwargs": {
                    "trust_remote_code": True,
                    "model_id": "sentence-transformers/all-minilm-l6-v2",
                    "provider": "CUDAExecutionProvider",
                    "use_auth_token": None,
                },
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self, mock_check_valid_model):  # noqa: ARG002
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-minilm-l6-v2",
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
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
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
        )
        embedder.warm_up()
        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OptimumDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OptimumDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self, mock_check_valid_model):  # noqa: ARG002
        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
        )
        embedder.warm_up()
        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    @pytest.mark.integration
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = OptimumDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            prefix="prefix ",
            suffix=" suffix",
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
            batch_size=1,
        )
        embedder.warm_up()

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 768
            assert all(isinstance(x, float) for x in doc.embedding)
