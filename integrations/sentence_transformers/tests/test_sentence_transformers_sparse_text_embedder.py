# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder import (
    SentenceTransformersSparseTextEmbedder,
)

TYPE_NAME = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder"


class TestSentenceTransformersSparseTextEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.trust_remote_code is False
        assert embedder.revision is None
        assert embedder.local_files_only is False

    def test_init_with_parameters(self):
        embedder = SentenceTransformersSparseTextEmbedder(
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
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.trust_remote_code is True
        assert embedder.revision == "v1.0"
        assert embedder.local_files_only is True

    def test_to_dict(self):
        component = SentenceTransformersSparseTextEmbedder(model="model", device=ComponentDevice.from_str("cpu"))
        data = component.to_dict()
        assert data == {
            "type": TYPE_NAME,
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "trust_remote_code": False,
                "revision": None,
                "local_files_only": False,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersSparseTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            trust_remote_code=True,
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float32},
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": False},
        )
        data = component.to_dict()
        assert data == {
            "type": TYPE_NAME,
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "prefix": "prefix",
                "suffix": "suffix",
                "trust_remote_code": True,
                "revision": None,
                "local_files_only": True,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
                "backend": "torch",
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersSparseTextEmbedder(model="model", token=Secret.from_token("fake-api-token"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    def test_from_dict(self):
        data = {
            "type": TYPE_NAME,
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "trust_remote_code": False,
                "revision": "v1.0",
                "local_files_only": False,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
            },
        }
        component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cpu")
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.trust_remote_code is False
        assert component.revision == "v1.0"
        assert component.local_files_only is False
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": False}

    def test_from_dict_no_default_parameters(self):
        data = {"type": TYPE_NAME, "init_parameters": {}}
        component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.model == "prithivida/Splade_PP_en_v2"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.trust_remote_code is False
        assert component.revision is None
        assert component.local_files_only is False

    def test_from_dict_none_device(self):
        data = {
            "type": TYPE_NAME,
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": None,
                "prefix": "",
                "suffix": "",
                "trust_remote_code": False,
                "local_files_only": False,
            },
        }
        component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.trust_remote_code is False
        assert component.revision is None
        assert component.local_files_only is False

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder._SentenceTransformersSparseEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersSparseTextEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs={"model_max_length": 512},
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
            model_kwargs=None,
            processor_kwargs={"model_max_length": 512},
            config_kwargs=None,
            backend="torch",
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder._SentenceTransformersSparseEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()

        def fake_embed(data, **kwargs):
            return [SparseEmbedding(indices=[1, 3], values=[0.5, 0.7]) for _ in range(len(data))]

        embedder.embedding_backend.embed = fake_embed

        text = "a nice text to embed"

        result = embedder.run(text=text)
        sparse_embedding = result["sparse_embedding"]

        assert isinstance(sparse_embedding, SparseEmbedding)
        assert sparse_embedding.indices == [1, 3]
        assert sparse_embedding.values == [0.5, 0.7]

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="SentenceTransformersSparseTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder._SentenceTransformersSparseEmbeddingBackendFactory"
    )
    def test_model_onnx_backend(self, mocked_factory):
        onnx_embedder = SentenceTransformersSparseTextEmbedder(
            model="prithivida/Splade_PP_en_v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={
                "file_name": "onnx/model.onnx"
            },  # setting the path isn't necessary if the repo contains a "onnx/model.onnx" file but this is to
            # prevent a HF warning
            backend="onnx",
        )
        onnx_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="prithivida/Splade_PP_en_v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            model_kwargs={"file_name": "onnx/model.onnx"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder._SentenceTransformersSparseEmbeddingBackendFactory"
    )
    def test_model_openvino_backend(self, mocked_factory):
        openvino_embedder = SentenceTransformersSparseTextEmbedder(
            model="prithivida/Splade_PP_en_v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={
                "file_name": "openvino/openvino_model.xml"
            },  # setting the path isn't necessary if the repo contains a "openvino/openvino_model.xml" file but
            # this is to prevent a HF warning
            backend="openvino",
        )
        openvino_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="prithivida/Splade_PP_en_v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    @patch(
        "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder._SentenceTransformersSparseEmbeddingBackendFactory"
    )
    @pytest.mark.parametrize("model_kwargs", [{"torch_dtype": "bfloat16"}, {"torch_dtype": "float16"}])
    def test_dtype_on_gpu(self, mocked_factory, model_kwargs):
        torch_dtype_embedder = SentenceTransformersSparseTextEmbedder(
            model="prithivida/Splade_PP_en_v2",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs=model_kwargs,
        )
        torch_dtype_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="prithivida/Splade_PP_en_v2",
            device="cuda:0",
            auth_token=None,
            trust_remote_code=False,
            revision=None,
            local_files_only=False,
            model_kwargs=model_kwargs,
            processor_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )

    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=10)
    def test_live_run_sparse_text_embedder(self, del_hf_env_vars_if_empty):

        text = "I love Nine Inch Nails"
        embedder = SentenceTransformersSparseTextEmbedder(
            model="sparse-encoder-testing/splade-bert-tiny-nq", device=ComponentDevice.from_str("cpu")
        )
        result = embedder.run(text=text)
        sparse_embedding = result["sparse_embedding"]

        assert isinstance(sparse_embedding, SparseEmbedding)
        assert isinstance(sparse_embedding.indices, list)
        assert isinstance(sparse_embedding.values, list)
        assert len(sparse_embedding.indices) == len(sparse_embedding.values)
        assert len(sparse_embedding.indices) > 0


_PROCESSOR_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_PROCESSOR_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_PROCESSOR_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."
_PROCESSOR_TYPE_NAME = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder"
_PROCESSOR_PATCH_TARGET = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder._SentenceTransformersSparseEmbeddingBackendFactory"


def _processor_util_warnings(caplog):
    return [
        record for record in caplog.records if record.name == _PROCESSOR_UTIL_LOGGER and record.levelname == "WARNING"
    ]


class TestSentenceTransformersSparseTextEmbedderProcessorKwargs:
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
            component = SentenceTransformersSparseTextEmbedder(
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
        component = SentenceTransformersSparseTextEmbedder(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in data["init_parameters"]
        legacy = SentenceTransformersSparseTextEmbedder(model="model", tokenizer_kwargs={"model_max_length": 128})
        assert set(data["init_parameters"]) == set(legacy.to_dict()["init_parameters"])

    def test_from_dict_accepts_canonical_key_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "processor_kwargs": {"model_max_length": 128}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_legacy_only_is_quiet(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "tokenizer_kwargs": {"model_max_length": 512}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSparseTextEmbedder.from_dict(data)
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
            component = SentenceTransformersSparseTextEmbedder.from_dict(data)
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
            component = SentenceTransformersSparseTextEmbedder.from_dict(data)
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
            component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {}
        assert len(_processor_util_warnings(caplog)) == 1

    def test_canonical_round_trip_is_quiet(self, caplog):
        component = SentenceTransformersSparseTextEmbedder(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            restored = SentenceTransformersSparseTextEmbedder.from_dict(data)
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
        embedder = SentenceTransformersSparseTextEmbedder(
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
            model_kwargs=None,
            processor_kwargs=expected,
            config_kwargs=None,
            backend="torch",
        )
        assert "tokenizer_kwargs" not in mocked_factory.get_embedding_backend.call_args.kwargs

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_repeated_does_not_reconstruct_or_warn(self, mocked_factory, caplog):
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            embedder = SentenceTransformersSparseTextEmbedder(model="model", tokenizer_kwargs={"model_max_length": 512})
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
        embedder = SentenceTransformersSparseTextEmbedder(
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
        embedder = SentenceTransformersSparseTextEmbedder(
            model="model", token=None, device=ComponentDevice.from_str("cpu")
        )
        embedder.tokenizer_kwargs = {"model_max_length": 128}
        embedder.warm_up()
        assert mocked_factory.get_embedding_backend.call_args.kwargs["processor_kwargs"] == {"model_max_length": 128}
        assert embedder.to_dict()["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}

    @pytest.mark.integration
    @pytest.mark.parametrize("kwargs_name", ["tokenizer_kwargs", "processor_kwargs"])
    def test_processor_kwargs_integration(self, kwargs_name, del_hf_env_vars_if_empty):
        embedder = SentenceTransformersSparseTextEmbedder(
            model="sparse-encoder-testing/splade-bert-tiny-nq",
            device=ComponentDevice.from_str("cpu"),
            **{kwargs_name: {"model_max_length": 128}},
        )
        embedder.warm_up()
        assert embedder.embedding_backend is not None
        assert embedder.embedding_backend.model.max_seq_length == 128
        result = embedder.run(text="I love Nine Inch Nails")
        sparse_embedding = result["sparse_embedding"]
        assert isinstance(sparse_embedding, SparseEmbedding)
        assert len(sparse_embedding.indices) == len(sparse_embedding.values)
        assert len(sparse_embedding.indices) > 0
