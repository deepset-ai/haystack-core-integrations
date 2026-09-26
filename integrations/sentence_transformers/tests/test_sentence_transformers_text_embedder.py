# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import MagicMock, patch

import pytest
import torch
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder import (
    SentenceTransformersTextEmbedder,
)


class TestSentenceTransformersTextEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False
        assert embedder.trust_remote_code is False
        assert embedder.revision is None
        assert embedder.local_files_only is False
        assert embedder.truncate_dim is None
        assert embedder.precision == "float32"
        assert embedder.quantization_ranges is None

    def test_init_with_parameters(self):
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            trust_remote_code=True,
            revision="v1.0",
            local_files_only=True,
            truncate_dim=256,
            precision="int8",
            quantization_ranges=[[-1.0, -1.0], [1.0, 1.0]],
        )
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True
        assert embedder.trust_remote_code is True
        assert embedder.revision == "v1.0"
        assert embedder.local_files_only is True
        assert embedder.truncate_dim == 256
        assert embedder.precision == "int8"
        assert embedder.quantization_ranges == [[-1.0, -1.0], [1.0, 1.0]]

    def test_init_quantized_precision_without_ranges_warns(self, caplog):
        with caplog.at_level("WARNING"):
            SentenceTransformersTextEmbedder(model="model", precision="int8")
        assert "quantization_ranges" in caplog.text

    def test_to_dict(self):
        component = SentenceTransformersTextEmbedder(model="model", device=ComponentDevice.from_str("cpu"))
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "revision": None,
                "local_files_only": False,
                "truncate_dim": None,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "encode_kwargs": None,
                "config_kwargs": None,
                "precision": "float32",
                "backend": "torch",
                "quantization_ranges": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            trust_remote_code=True,
            local_files_only=True,
            truncate_dim=256,
            model_kwargs={"torch_dtype": torch.float32},
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": False},
            precision="int8",
            encode_kwargs={"task": "clustering"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
                "trust_remote_code": True,
                "revision": None,
                "local_files_only": True,
                "truncate_dim": 256,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
                "precision": "int8",
                "encode_kwargs": {"task": "clustering"},
                "backend": "torch",
                "quantization_ranges": None,
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersTextEmbedder(model="model", token=Secret.from_token("fake-api-token"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "revision": "v1.0",
                "local_files_only": False,
                "truncate_dim": None,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
                "precision": "float32",
            },
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cpu")
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False
        assert component.trust_remote_code is False
        assert component.revision == "v1.0"
        assert component.local_files_only is False
        assert component.truncate_dim is None
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": False}
        assert component.precision == "float32"

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {},
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "sentence-transformers/all-mpnet-base-v2"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False
        assert component.trust_remote_code is False
        assert component.revision is None
        assert component.local_files_only is False
        assert component.truncate_dim is None
        assert component.precision == "float32"

    def test_from_dict_none_device(self):
        data = {
            "type": "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "local_files_only": False,
                "truncate_dim": 256,
                "precision": "int8",
            },
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False
        assert component.trust_remote_code is False
        assert component.revision is None
        assert component.local_files_only is False
        assert component.truncate_dim == 256
        assert component.precision == "int8"

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs={"model_max_length": 512},
        )
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.embedding_backend.model.max_seq_length = 512
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs=None,
            processor_kwargs={"model_max_length": 512},
            config_kwargs=None,
            backend="torch",
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **_: [[random.random() for _ in range(16)] for _ in range(len(x))]

        text = "a nice text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(el, float) for el in embedding)

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="SentenceTransformersTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    def test_embed_encode_kwargs(self):
        embedder = SentenceTransformersTextEmbedder(model="model", encode_kwargs={"task": "retrieval.query"})
        embedder.embedding_backend = MagicMock()
        text = "a nice text to embed"
        embedder.run(text=text)
        embedder.embedding_backend.embed.assert_called_once_with(
            [text],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
            precision="float32",
            quantization_ranges=None,
            task="retrieval.query",
        )

    def test_run_with_quantization_ranges(self):
        ranges = [[-1.0, -1.0], [1.0, 1.0]]
        embedder = SentenceTransformersTextEmbedder(model="model", precision="int8", quantization_ranges=ranges)
        embedder.embedding_backend = MagicMock()
        text = "a nice text to embed"
        embedder.run(text=text)
        embedder.embedding_backend.embed.assert_called_once_with(
            [text],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
            precision="int8",
            quantization_ranges=ranges,
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_model_onnx_backend(self, mocked_factory):
        onnx_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            # setting the path isn't necessary if the repo contains a "onnx/model.onnx" file but this is to prevent
            # a HF warning
            model_kwargs={"file_name": "onnx/model.onnx"},
            backend="onnx",
        )
        onnx_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs={"file_name": "onnx/model.onnx"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_model_openvino_backend(self, mocked_factory):
        openvino_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            # setting the path isn't necessary if the repo contains a "openvino/openvino_model.xml" file but
            # this is to prevent a HF warning
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            backend="openvino",
        )
        openvino_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    @pytest.mark.parametrize("model_kwargs", [{"torch_dtype": "bfloat16"}, {"torch_dtype": "float16"}])
    def test_dtype_on_gpu(self, mocked_factory, model_kwargs):
        torch_dtype_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs=model_kwargs,
        )
        torch_dtype_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda:0",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs=model_kwargs,
            processor_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )

    @pytest.mark.integration
    def test_run_trunc(self, del_hf_env_vars_if_empty):
        """
        sentence-transformers-testing/stsb-bert-tiny-safetensors maps sentences & paragraphs to a 128 dimensional dense
        vector space
        """
        checkpoint = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        text = "a nice text to embed"

        embedder_trunc = SentenceTransformersTextEmbedder(
            model=checkpoint, truncate_dim=64, device=ComponentDevice.from_str("cpu")
        )
        result_trunc = embedder_trunc.run(text=text)
        embedding_trunc = result_trunc["embedding"]

        # without truncation, the embedding has 128 length
        assert len(embedding_trunc) == 64

    @pytest.mark.integration
    def test_run_quantization(self, del_hf_env_vars_if_empty):
        """
        sentence-transformers-testing/stsb-bert-tiny-safetensors maps sentences & paragraphs to a 128 dimensional dense
        vector space
        """
        checkpoint = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        text = "a nice text to embed"

        embedder_def = SentenceTransformersTextEmbedder(
            model=checkpoint, precision="int8", device=ComponentDevice.from_str("cpu")
        )
        result_def = embedder_def.run(text=text)
        embedding_def = result_def["embedding"]

        assert len(embedding_def) == 128
        assert all(isinstance(el, int) for el in embedding_def)

    @pytest.mark.integration
    def test_run_quantization_with_ranges(self, del_hf_env_vars_if_empty):
        """
        sentence-transformers-testing/stsb-bert-tiny-safetensors maps sentences & paragraphs to a 128 dimensional dense
        vector space
        """
        checkpoint = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        text = "a nice text to embed"
        ranges = [[-1.0] * 128, [1.0] * 128]

        embedder = SentenceTransformersTextEmbedder(model=checkpoint, precision="int8", quantization_ranges=ranges)
        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert len(embedding) == 128
        assert all(isinstance(el, int) for el in embedding)
        # without explicit ranges, a single text produces a degenerate all-equal embedding
        assert len(set(embedding)) > 1


_PROCESSOR_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_PROCESSOR_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_PROCESSOR_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."
_PROCESSOR_TYPE_NAME = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder"
_PROCESSOR_PATCH_TARGET = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"


def _processor_util_warnings(caplog):
    return [
        record for record in caplog.records if record.name == _PROCESSOR_UTIL_LOGGER and record.levelname == "WARNING"
    ]


class TestSentenceTransformersTextEmbedderProcessorKwargs:
    @pytest.mark.parametrize(
        ("legacy", "canonical", "expect_warn", "expect_conflict"),
        [
            pytest.param(None, None, False, False, id="both-none"),
            pytest.param({"model_max_length": 512}, None, True, False, id="legacy-only"),
            pytest.param(None, {"model_max_length": 128}, False, False, id="canonical-only"),
            pytest.param({}, None, True, False, id="legacy-empty"),
            pytest.param(None, {}, False, False, id="canonical-empty"),
            pytest.param({"model_max_length": 512}, {}, True, True, id="legacy-plus-canonical-empty"),
            pytest.param({"model_max_length": 512}, {"model_max_length": 128}, True, True, id="both-nonempty"),
            pytest.param({"model_max_length": 512}, None, True, False, id="legacy-plus-explicit-canonical-none"),
        ],
    )
    def test_constructor_matrix(self, caplog, legacy, canonical, expect_warn, expect_conflict):
        legacy_snapshot = None if legacy is None else dict(legacy)
        canonical_snapshot = None if canonical is None else dict(canonical)
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersTextEmbedder(
                model="model",
                token=None,
                device=ComponentDevice.from_str("cpu"),
                tokenizer_kwargs=legacy,
                processor_kwargs=canonical,
            )
        expected = canonical if canonical is not None else legacy
        assert component.tokenizer_kwargs == expected
        if expected is None:
            assert component.tokenizer_kwargs is None
        elif canonical is not None:
            assert component.tokenizer_kwargs is canonical
        else:
            assert component.tokenizer_kwargs is legacy
        assert not hasattr(component, "processor_kwargs")
        assert legacy == legacy_snapshot
        assert canonical == canonical_snapshot
        records = _processor_util_warnings(caplog)
        if expect_warn:
            assert len(records) == 1
            expected_message = _PROCESSOR_DEPRECATED_MESSAGE
            if expect_conflict:
                expected_message += _PROCESSOR_CONFLICT_SUFFIX
            assert records[0].getMessage() == expected_message
        else:
            assert records == []

    def test_to_dict_emits_legacy_key_only(self):
        component = SentenceTransformersTextEmbedder(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in data["init_parameters"]
        legacy = SentenceTransformersTextEmbedder(model="model", tokenizer_kwargs={"model_max_length": 128})
        assert set(data["init_parameters"]) == set(legacy.to_dict()["init_parameters"])

    def test_from_dict_accepts_canonical_key_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "processor_kwargs": {"model_max_length": 128}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_legacy_only_is_quiet(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "tokenizer_kwargs": {"model_max_length": 512}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_both_keys_canonical_wins_with_warning(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": {"model_max_length": 128},
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        records = _processor_util_warnings(caplog)
        assert len(records) == 1
        assert records[0].getMessage() == _PROCESSOR_DEPRECATED_MESSAGE + _PROCESSOR_CONFLICT_SUFFIX
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 512}
        assert data["init_parameters"]["processor_kwargs"] == {"model_max_length": 128}
        reserialized = component.to_dict()
        assert reserialized["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in reserialized["init_parameters"]

    def test_from_dict_explicit_canonical_none_falls_back_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": None,
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_canonical_empty_overrides_legacy(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": {},
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {}
        assert len(_processor_util_warnings(caplog)) == 1

    def test_canonical_round_trip_is_quiet(self, caplog):
        component = SentenceTransformersTextEmbedder(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            restored = SentenceTransformersTextEmbedder.from_dict(data)
        assert restored.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    @pytest.mark.parametrize(
        ("legacy", "canonical", "expected"),
        [
            pytest.param(None, None, None, id="both-none"),
            pytest.param({"model_max_length": 512}, None, {"model_max_length": 512}, id="legacy-only"),
            pytest.param(None, {"model_max_length": 128}, {"model_max_length": 128}, id="canonical-only"),
            pytest.param(
                {"model_max_length": 512}, {"model_max_length": 128}, {"model_max_length": 128}, id="both-nonempty"
            ),
            pytest.param({"model_max_length": 512}, {}, {}, id="canonical-empty-wins"),
        ],
    )
    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_forwards_effective_as_canonical(self, mocked_factory, legacy, canonical, expected):
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs=legacy,
            processor_kwargs=canonical,
        )
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs=None,
            processor_kwargs=expected,
            config_kwargs=None,
            backend="torch",
        )
        assert "tokenizer_kwargs" not in mocked_factory.get_embedding_backend.call_args.kwargs

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_repeated_does_not_reconstruct_or_warn(self, mocked_factory, caplog):
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            embedder = SentenceTransformersTextEmbedder(model="model", tokenizer_kwargs={"model_max_length": 512})
            embedder.warm_up()
            embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()
        assert len(_processor_util_warnings(caplog)) == 1

    @pytest.mark.parametrize(
        ("legacy", "canonical", "expected_length"),
        [
            pytest.param({"model_max_length": 128}, None, 128, id="legacy-length"),
            pytest.param(None, {"model_max_length": 128}, 128, id="canonical-length"),
            pytest.param({"model_max_length": 512}, {"model_max_length": 128}, 128, id="canonical-override"),
            pytest.param(None, None, None, id="both-none"),
            pytest.param({}, None, None, id="legacy-empty"),
            pytest.param({"model_max_length": 0}, None, None, id="legacy-zero"),
            pytest.param({"other": "value"}, None, None, id="legacy-without-length"),
            pytest.param({"model_max_length": 512}, {}, None, id="canonical-empty-suppresses"),
        ],
    )
    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_model_max_length_sync(self, mocked_factory, legacy, canonical, expected_length):
        mocked_factory.get_embedding_backend.return_value.model.max_seq_length = 999
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs=legacy,
            processor_kwargs=canonical,
        )
        embedder.warm_up()
        if expected_length is None:
            assert mocked_factory.get_embedding_backend.return_value.model.max_seq_length == 999
        else:
            assert mocked_factory.get_embedding_backend.return_value.model.max_seq_length == expected_length

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_legacy_reassignment_before_warmup_is_effective(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model="model", token=None, device=ComponentDevice.from_str("cpu"))
        embedder.tokenizer_kwargs = {"model_max_length": 128}
        embedder.warm_up()
        assert mocked_factory.get_embedding_backend.call_args.kwargs["processor_kwargs"] == {"model_max_length": 128}
        assert embedder.to_dict()["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}

    @pytest.mark.integration
    @pytest.mark.parametrize("kwargs_name", ["tokenizer_kwargs", "processor_kwargs"])
    def test_processor_kwargs_integration(self, kwargs_name, del_hf_env_vars_if_empty):
        checkpoint = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
        embedder = SentenceTransformersTextEmbedder(
            model=checkpoint, device=ComponentDevice.from_str("cpu"), **{kwargs_name: {"model_max_length": 128}}
        )
        embedder.warm_up()
        assert embedder.embedding_backend is not None
        assert embedder.embedding_backend.model.max_seq_length == 128
        result = embedder.run(text="a nice text to embed")
        assert len(result["embedding"]) == 128
