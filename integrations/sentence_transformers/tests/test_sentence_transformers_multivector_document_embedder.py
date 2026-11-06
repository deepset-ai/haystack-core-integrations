# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from haystack import Document
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_document_embedder import (
    SentenceTransformersMultivectorDocumentEmbedder,
)


class TestSentenceTransformersMultivectorDocumentEmbedder:
    def test_init_with_parameters(self):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            trust_remote_code=True,
            revision="v1.0",
            local_files_only=True,
        )
        assert embedder.model == "model"
        assert embedder.embedding_backend is None
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder.trust_remote_code
        assert embedder.revision == "v1.0"
        assert embedder.local_files_only

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_init_invalid_batch_size(self, batch_size):
        with pytest.raises(ValueError, match="batch_size must be greater than zero"):
            SentenceTransformersMultivectorDocumentEmbedder(model="model", batch_size=batch_size)

    @pytest.mark.parametrize(
        "model_kwargs, expected_model_kwargs",
        [(None, None), ({"torch_dtype": torch.float32}, {"torch_dtype": "torch.float32"})],
    )
    def test_to_dict(self, model_kwargs, expected_model_kwargs):
        component = SentenceTransformersMultivectorDocumentEmbedder(
            model="model", device=ComponentDevice.from_str("cpu"), model_kwargs=model_kwargs
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_document_embedder.SentenceTransformersMultivectorDocumentEmbedder",
            "init_parameters": {
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "embedding_separator": "\n",
                "meta_fields_to_embed": [],
                "trust_remote_code": False,
                "revision": None,
                "local_files_only": False,
                "model_kwargs": expected_model_kwargs,
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
            },
        }

    @pytest.mark.parametrize(
        "device, model_kwargs, expected_model_kwargs",
        [
            (None, None, None),
            (ComponentDevice.from_str("cuda:0"), {"torch_dtype": "torch.float32"}, {"torch_dtype": torch.float32}),
        ],
    )
    def test_from_dict(self, device, model_kwargs, expected_model_kwargs):
        init_parameters = {
            "model": "model",
            "device": device.to_dict() if device is not None else None,
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
            "prefix": "prefix",
            "suffix": "suffix",
            "batch_size": 64,
            "progress_bar": False,
            "embedding_separator": " - ",
            "meta_fields_to_embed": ["meta_field"],
            "trust_remote_code": True,
            "revision": "v1.0",
            "local_files_only": True,
            "model_kwargs": model_kwargs.copy() if model_kwargs is not None else None,
            "tokenizer_kwargs": {"model_max_length": 512},
            "config_kwargs": {"use_memory_efficient_attention": True},
        }
        component = SentenceTransformersMultivectorDocumentEmbedder.from_dict(
            {
                "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_document_embedder.SentenceTransformersMultivectorDocumentEmbedder",
                "init_parameters": init_parameters,
            }
        )
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(device)
        assert component.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert component.prefix == "prefix"
        assert component.suffix == "suffix"
        assert component.batch_size == 64
        assert component.progress_bar is False
        assert component.embedding_separator == " - "
        assert component.trust_remote_code
        assert component.revision == "v1.0"
        assert component.local_files_only
        assert component.meta_fields_to_embed == ["meta_field"]
        assert component.model_kwargs == expected_model_kwargs
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": True}

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_document_embedder._SentenceTransformersMultivectorEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs={"torch_dtype": "float16"},
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": True},
        )
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cuda:0",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            model_kwargs={"torch_dtype": "float16"},
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": True},
            backend="torch",
        )

    @pytest.mark.parametrize(
        "backend, file_name",
        [("onnx", "onnx/model.onnx"), ("openvino", "openvino/openvino_model.xml")],
    )
    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_document_embedder._SentenceTransformersMultivectorEmbeddingBackendFactory"
    )
    def test_model_backend(self, mocked_factory, backend, file_name):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(
            model="answerdotai/answerai-colbert-small-v1",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": file_name},
            backend=backend,
        )
        embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="answerdotai/answerai-colbert-small-v1",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            model_kwargs={"file_name": file_name},
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend=backend,
        )

    @pytest.mark.parametrize("content, expected_text", [("document", "document"), (None, ""), ("", "")])
    def test_run(self, content, expected_text):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(model="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed.return_value = [[[0.1, 0.2], [0.3, 0.4]] for _ in range(2)]

        documents = [Document(content=content) for _ in range(2)]

        result = embedder.run(documents=documents)
        embedder.embedding_backend.embed.assert_called_once_with(
            data=[expected_text] * 2, is_query=False, batch_size=32, show_progress_bar=True
        )

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"], strict=True):
            assert new_doc is not doc
            assert doc.multivector_embedding is None
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.multivector_embedding, list)
            assert new_doc.multivector_embedding == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.parametrize("documents", [None, [1, 2, 3], [Document(content="valid"), "invalid"]])
    def test_run_wrong_input_format(self, documents):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(model="model")
        with pytest.raises(TypeError, match="expects a list of Documents as input"):
            embedder.run(documents=documents)

    def test_run_empty_documents(self):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(model="model")
        assert embedder.run(documents=[]) == {"documents": []}
        assert embedder.embedding_backend is None

    @pytest.mark.parametrize("embedding_count", [0, 2])
    def test_run_embedding_count_mismatch(self, embedding_count):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(model="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed.return_value = [[[0.1, 0.2]]] * embedding_count
        with pytest.raises(ValueError, match="zip"):
            embedder.run(documents=[Document(content="document")])

    @pytest.mark.parametrize(
        "meta_fields_to_embed, meta, expected_text",
        [
            (["meta_field"], {"meta_field": "meta_value"}, "meta_value\ndocument"),
            (
                ["rating", "is_available", "missing", "none_value"],
                {"rating": 0, "is_available": False, "none_value": None},
                "0\nFalse\ndocument",
            ),
        ],
        ids=["text", "falsy-and-missing-values"],
    )
    def test_embed_metadata(self, meta_fields_to_embed, meta, expected_text):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(
            model="model", meta_fields_to_embed=meta_fields_to_embed, embedding_separator="\n"
        )
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed.return_value = [[[0.1, 0.2], [0.3, 0.4]]]
        embedder.run(documents=[Document(content="document", meta=meta)])
        embedder.embedding_backend.embed.assert_called_once_with(
            data=[expected_text], is_query=False, batch_size=32, show_progress_bar=True
        )

    def test_prefix_suffix(self):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(
            model="model",
            prefix="my_prefix ",
            suffix=" my_suffix",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="\n",
        )
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed.return_value = [[[0.1, 0.2], [0.3, 0.4]] for _ in range(5)]
        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]
        embedder.run(documents=documents)
        embedder.embedding_backend.embed.assert_called_once_with(
            data=[
                "my_prefix meta_value 0\ndocument number 0 my_suffix",
                "my_prefix meta_value 1\ndocument number 1 my_suffix",
                "my_prefix meta_value 2\ndocument number 2 my_suffix",
                "my_prefix meta_value 3\ndocument number 3 my_suffix",
                "my_prefix meta_value 4\ndocument number 4 my_suffix",
            ],
            is_query=False,
            batch_size=32,
            show_progress_bar=True,
        )

    @pytest.mark.integration
    def test_live_run(self, del_hf_env_vars_if_empty):
        embedder = SentenceTransformersMultivectorDocumentEmbedder(
            model="answerdotai/answerai-colbert-small-v1",
            device=ComponentDevice.from_str("cpu"),
            batch_size=1,
            progress_bar=False,
        )
        documents = [Document(content="Paris is the capital of France."), Document(content="Bananas are yellow.")]
        result = embedder.run(documents=documents)
        assert len(result["documents"]) == len(documents)
        for document, embedded_document in zip(documents, result["documents"], strict=True):
            assert embedded_document is not document
            assert document.multivector_embedding is None
            embedding = embedded_document.multivector_embedding
            assert isinstance(embedding, list)
            assert len(embedding) > 1
            assert all(len(vector) == 96 for vector in embedding)
            assert all(isinstance(value, float) for vector in embedding for value in vector)
