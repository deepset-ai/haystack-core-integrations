# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_text_embedder import (
    SentenceTransformersMultivectorTextEmbedder,
)


class TestSentenceTransformersMultivectorTextEmbedder:
    def test_init_with_parameters(self):
        embedder = SentenceTransformersMultivectorTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
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
        assert embedder.trust_remote_code is True
        assert embedder.revision == "v1.0"
        assert embedder.local_files_only is True

    @pytest.mark.parametrize(
        "model_kwargs, expected_model_kwargs",
        [(None, None), ({"torch_dtype": torch.float32}, {"torch_dtype": "torch.float32"})],
    )
    def test_to_dict(self, model_kwargs, expected_model_kwargs):
        component = SentenceTransformersMultivectorTextEmbedder(
            model="model", device=ComponentDevice.from_str("cpu"), model_kwargs=model_kwargs
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_text_embedder.SentenceTransformersMultivectorTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "trust_remote_code": False,
                "revision": None,
                "local_files_only": False,
                "model_kwargs": expected_model_kwargs,
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersMultivectorTextEmbedder(
            model="model", token=Secret.from_token("fake-api-token")
        )
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    @pytest.mark.parametrize(
        "device, model_kwargs, expected_model_kwargs",
        [
            (None, None, None),
            (ComponentDevice.from_str("cpu"), {"torch_dtype": "torch.float32"}, {"torch_dtype": torch.float32}),
        ],
    )
    def test_from_dict(self, device, model_kwargs, expected_model_kwargs):
        data = {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_text_embedder.SentenceTransformersMultivectorTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": device.to_dict() if device is not None else None,
                "prefix": "",
                "suffix": "",
                "trust_remote_code": False,
                "revision": "v1.0",
                "local_files_only": False,
                "model_kwargs": model_kwargs.copy() if model_kwargs is not None else None,
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
            },
        }
        component = SentenceTransformersMultivectorTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(device)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.trust_remote_code is False
        assert component.revision == "v1.0"
        assert component.local_files_only is False
        assert component.model_kwargs == expected_model_kwargs
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": False}

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_text_embedder._SentenceTransformersMultivectorEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersMultivectorTextEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs={"torch_dtype": "float16"},
            tokenizer_kwargs={"model_max_length": 512},
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
            config_kwargs=None,
            backend="torch",
        )

    @pytest.mark.parametrize(
        "backend, file_name",
        [("onnx", "onnx/model.onnx"), ("openvino", "openvino/openvino_model.xml")],
    )
    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_multivector_text_embedder._SentenceTransformersMultivectorEmbeddingBackendFactory"
    )
    def test_model_backend(self, mocked_factory, backend, file_name):
        embedder = SentenceTransformersMultivectorTextEmbedder(
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

    def test_run(self):
        embedder = SentenceTransformersMultivectorTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda data, **_: [[[0.1, 0.2], [0.3, 0.4]] for _ in data]

        text = "a nice text to embed"

        result = embedder.run(text=text)
        embedding = result["multivector_embedding"]

        assert isinstance(embedding, list)
        assert embedding == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.parametrize("text", [None, [1, 2, 3]])
    def test_run_wrong_input_format(self, text):
        embedder = SentenceTransformersMultivectorTextEmbedder(model="model")
        with pytest.raises(TypeError, match="expects a string as input"):
            embedder.run(text=text)

    def test_prefix_suffix(self):
        embedder = SentenceTransformersMultivectorTextEmbedder(model="model", prefix="my_prefix ", suffix=" my_suffix")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed.return_value = [[[0.1, 0.2], [0.3, 0.4]]]
        embedder.run(text="text")
        embedder.embedding_backend.embed.assert_called_once_with(data=["my_prefix text my_suffix"], is_query=True)

    @pytest.mark.integration
    def test_live_run(self, del_hf_env_vars_if_empty):
        embedder = SentenceTransformersMultivectorTextEmbedder(
            model="answerdotai/answerai-colbert-small-v1", device=ComponentDevice.from_str("cpu")
        )
        result = embedder.run(text="What is the capital of France?")
        embedding = result["multivector_embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 1
        assert all(len(vector) == 96 for vector in embedding)
        assert all(isinstance(value, float) for vector in embedding for value in vector)
