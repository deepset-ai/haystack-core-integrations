# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend import (
    _SentenceTransformersEmbeddingBackend,
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack_integrations.utils.sentence_transformers import (
    _normalize_processor_kwargs,
    _resolve_processor_kwargs,
)


@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.SentenceTransformer")
def test_factory_behavior(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    same_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    another_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="another_model", device="cpu"
    )
    yet_another_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu", trust_remote_code=True
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend
    assert yet_another_embedding_backend is not embedding_backend


@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.SentenceTransformer")
def test_model_initialization(mock_sentence_transformer):
    _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="model",
        device="cpu",
        auth_token=Secret.from_token("fake-api-token"),
        trust_remote_code=True,
        local_files_only=True,
        truncate_dim=256,
        backend="torch",
    )
    mock_sentence_transformer.assert_called_once_with(
        model_name_or_path="model",
        device="cpu",
        token="fake-api-token",
        trust_remote_code=True,
        revision=None,
        local_files_only=True,
        truncate_dim=256,
        model_kwargs=None,
        processor_kwargs=None,
        config_kwargs=None,
        backend="torch",
    )


@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.SentenceTransformer")
def test_embedding_function_with_kwargs(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model="model")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data, normalize_embeddings=True)

    embedding_backend.model.encode.assert_called_once_with(data, normalize_embeddings=True)


@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.quantize_embeddings")
@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.SentenceTransformer")
def test_embedding_function_with_quantization_ranges(mock_sentence_transformer, mock_quantize_embeddings):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model="quantized_model")
    embedding_backend.model.encode.return_value = np.array([[0.1, 0.2]])
    mock_quantize_embeddings.return_value = np.array([[12, 34]], dtype=np.int8)

    data = ["sentence"]
    ranges = [[-1.0, -1.0], [1.0, 1.0]]
    result = embedding_backend.embed(data=data, precision="int8", quantization_ranges=ranges)

    embedding_backend.model.encode.assert_called_once_with(data, precision="float32")
    assert mock_quantize_embeddings.call_count == 1
    _, called_kwargs = mock_quantize_embeddings.call_args
    assert called_kwargs["precision"] == "int8"
    assert np.array_equal(called_kwargs["ranges"], np.asarray(ranges))
    assert result == [[12, 34]]


@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.quantize_embeddings")
@patch("haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.SentenceTransformer")
def test_embedding_function_without_quantization_ranges(mock_sentence_transformer, mock_quantize_embeddings):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model="another_quantized")
    embedding_backend.model.encode.return_value = np.array([[1, 2]], dtype=np.int8)

    data = ["sentence"]
    embedding_backend.embed(data=data, precision="int8", quantization_ranges=None)

    embedding_backend.model.encode.assert_called_once_with(data, precision="int8")
    mock_quantize_embeddings.assert_not_called()


_PATCH_TARGET = (
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.backend.SentenceTransformer"
)
_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."


def _util_warnings(caplog):
    return [record for record in caplog.records if record.name == _UTIL_LOGGER and record.levelname == "WARNING"]


def test_resolve_processor_kwargs_both_none_is_quiet(caplog):
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        assert _resolve_processor_kwargs(tokenizer_kwargs=None, processor_kwargs=None) is None
    assert _util_warnings(caplog) == []


def test_resolve_processor_kwargs_legacy_only_warns_and_returns_same_object(caplog):
    legacy = {"model_max_length": 128}
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        resolved = _resolve_processor_kwargs(tokenizer_kwargs=legacy, processor_kwargs=None)
    assert resolved is legacy
    records = _util_warnings(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == _DEPRECATED_MESSAGE


def test_resolve_processor_kwargs_legacy_empty_dict_warns_and_wins(caplog):
    legacy = {}
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        resolved = _resolve_processor_kwargs(tokenizer_kwargs=legacy, processor_kwargs=None)
    assert resolved is legacy
    records = _util_warnings(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == _DEPRECATED_MESSAGE


def test_resolve_processor_kwargs_explicit_canonical_none_with_legacy_warns_without_conflict(caplog):
    legacy = {"model_max_length": 128}
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        resolved = _resolve_processor_kwargs(tokenizer_kwargs=legacy, processor_kwargs=None)
    assert resolved is legacy
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE]


@pytest.mark.parametrize("canonical", [{"model_max_length": 256}, {}], ids=["nonempty", "empty"])
def test_resolve_processor_kwargs_canonical_only_is_quiet(caplog, canonical):
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        resolved = _resolve_processor_kwargs(tokenizer_kwargs=None, processor_kwargs=canonical)
    assert resolved is canonical
    assert _util_warnings(caplog) == []


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ({"model_max_length": 128}, {"model_max_length": 256}),
        ({"model_max_length": 128}, {}),
    ],
    ids=["both-nonempty", "canonical-empty"],
)
def test_resolve_processor_kwargs_both_non_none_canonical_wins_with_conflict_log(caplog, legacy, canonical):
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        resolved = _resolve_processor_kwargs(tokenizer_kwargs=legacy, processor_kwargs=canonical)
    assert resolved is canonical
    records = _util_warnings(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == _DEPRECATED_MESSAGE + _CONFLICT_SUFFIX


def test_resolve_processor_kwargs_does_not_mutate_inputs(caplog):
    legacy = {"model_max_length": 128}
    canonical = {"model_max_length": 256}
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        _resolve_processor_kwargs(tokenizer_kwargs=legacy, processor_kwargs=canonical)
    assert legacy == {"model_max_length": 128}
    assert canonical == {"model_max_length": 256}


def test_resolve_processor_kwargs_warn_false_is_silent(caplog):
    legacy = {"model_max_length": 128}
    canonical = {"model_max_length": 256}
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        resolved = _resolve_processor_kwargs(tokenizer_kwargs=legacy, processor_kwargs=canonical, warn=False)
    assert resolved is canonical
    assert _util_warnings(caplog) == []


def test_normalize_processor_kwargs_absent_keys_returns_equal_copy():
    data = {"type": "SomeComponent", "init_parameters": {"model": "m", "device": "cpu"}}
    result = _normalize_processor_kwargs(data)
    assert result == data
    assert result is not data
    assert result["init_parameters"] is not data["init_parameters"]


def test_normalize_processor_kwargs_without_init_parameters_returns_equal_copy():
    data = {"type": "SomeComponent"}
    result = _normalize_processor_kwargs(data)
    assert result == data
    assert result is not data


def test_normalize_processor_kwargs_legacy_only_translates_silently(caplog):
    data = {
        "type": "SomeComponent",
        "init_parameters": {"model": "m", "tokenizer_kwargs": {"model_max_length": 128}},
    }
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        result = _normalize_processor_kwargs(data)
    assert result == {
        "type": "SomeComponent",
        "init_parameters": {"model": "m", "processor_kwargs": {"model_max_length": 128}},
    }
    assert _util_warnings(caplog) == []
    assert data == {
        "type": "SomeComponent",
        "init_parameters": {"model": "m", "tokenizer_kwargs": {"model_max_length": 128}},
    }


def test_normalize_processor_kwargs_legacy_only_none_translates_silently(caplog):
    data = {"type": "SomeComponent", "init_parameters": {"model": "m", "tokenizer_kwargs": None}}
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        result = _normalize_processor_kwargs(data)
    assert result == {"type": "SomeComponent", "init_parameters": {"model": "m", "processor_kwargs": None}}
    assert _util_warnings(caplog) == []
    assert set(data["init_parameters"]) == {"model", "tokenizer_kwargs"}


def test_normalize_processor_kwargs_canonical_only_keeps_content_silently(caplog):
    data = {
        "type": "SomeComponent",
        "init_parameters": {"model": "m", "processor_kwargs": {"model_max_length": 256}},
    }
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        result = _normalize_processor_kwargs(data)
    assert result == data
    assert result is not data
    assert _util_warnings(caplog) == []
    assert set(data["init_parameters"]) == {"model", "processor_kwargs"}


def test_normalize_processor_kwargs_both_non_none_canonical_wins_with_conflict_log(caplog):
    data = {
        "type": "SomeComponent",
        "init_parameters": {
            "model": "m",
            "tokenizer_kwargs": {"model_max_length": 128},
            "processor_kwargs": {"model_max_length": 256},
        },
    }
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        result = _normalize_processor_kwargs(data)
    assert result == {
        "type": "SomeComponent",
        "init_parameters": {"model": "m", "processor_kwargs": {"model_max_length": 256}},
    }
    records = _util_warnings(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == _DEPRECATED_MESSAGE + _CONFLICT_SUFFIX
    assert set(data["init_parameters"]) == {"model", "tokenizer_kwargs", "processor_kwargs"}


def test_normalize_processor_kwargs_canonical_empty_dict_wins_over_legacy(caplog):
    data = {
        "type": "SomeComponent",
        "init_parameters": {
            "tokenizer_kwargs": {"model_max_length": 128},
            "processor_kwargs": {},
        },
    }
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        result = _normalize_processor_kwargs(data)
    assert result["init_parameters"]["processor_kwargs"] == {}
    assert "tokenizer_kwargs" not in result["init_parameters"]
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE + _CONFLICT_SUFFIX]


def test_normalize_processor_kwargs_explicit_canonical_none_falls_back_to_legacy_silently(caplog):
    data = {
        "type": "SomeComponent",
        "init_parameters": {
            "tokenizer_kwargs": {"model_max_length": 128},
            "processor_kwargs": None,
        },
    }
    with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
        result = _normalize_processor_kwargs(data)
    assert result["init_parameters"]["processor_kwargs"] == {"model_max_length": 128}
    assert "tokenizer_kwargs" not in result["init_parameters"]
    assert _util_warnings(caplog) == []


def test_normalize_processor_kwargs_preserves_unrelated_and_nested_keys():
    encode_kwargs = {"normalize_embeddings": True}
    data = {
        "type": "SomeComponent",
        "init_parameters": {
            "model": "m",
            "encode_kwargs": encode_kwargs,
            "tokenizer_kwargs": {"model_max_length": 128},
        },
    }
    result = _normalize_processor_kwargs(data)
    assert result["init_parameters"]["model"] == "m"
    assert result["init_parameters"]["encode_kwargs"] == {"normalize_embeddings": True}
    # shallow copy: nested mappings are shared, outer mappings are new objects
    assert result["init_parameters"]["encode_kwargs"] is encode_kwargs
    assert result["init_parameters"] is not data["init_parameters"]
    assert set(data["init_parameters"]) == {"model", "encode_kwargs", "tokenizer_kwargs"}


@pytest.mark.parametrize(
    ("legacy", "canonical", "expected"),
    [
        (None, None, None),
        ({"model_max_length": 128}, None, {"model_max_length": 128}),
        ({}, None, {}),
        (None, {"model_max_length": 256}, {"model_max_length": 256}),
        (None, {}, {}),
        ({"model_max_length": 128}, {"model_max_length": 256}, {"model_max_length": 256}),
        ({"model_max_length": 128}, {}, {}),
    ],
    ids=[
        "none-none",
        "legacy-only",
        "legacy-empty-only",
        "canonical-only",
        "canonical-empty-only",
        "both-different",
        "canonical-empty-overrides-legacy",
    ],
)
def test_factory_processor_kwargs_matrix(monkeypatch, legacy, canonical, expected):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    legacy_snapshot = dict(legacy) if legacy is not None else None
    canonical_snapshot = dict(canonical) if canonical is not None else None
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="matrix_dense_model", tokenizer_kwargs=legacy, processor_kwargs=canonical
        )
    _, kwargs = mock_sentence_transformer.call_args
    assert kwargs["processor_kwargs"] == expected
    assert "tokenizer_kwargs" not in kwargs
    assert legacy == legacy_snapshot
    assert canonical == canonical_snapshot


def test_factory_explicit_canonical_none_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="explicit_none_dense_model",
            tokenizer_kwargs={"model_max_length": 128},
            processor_kwargs=None,
        )
    _, kwargs = mock_sentence_transformer.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 128}
    assert "tokenizer_kwargs" not in kwargs


def test_factory_alias_spellings_share_cache_identity(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        legacy_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="alias_dense_model", tokenizer_kwargs={"model_max_length": 128}
        )
        canonical_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="alias_dense_model", processor_kwargs={"model_max_length": 128}
        )
        both_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="alias_dense_model",
            tokenizer_kwargs={"other": 1},
            processor_kwargs={"model_max_length": 128},
        )
    assert canonical_backend is legacy_backend
    assert both_backend is legacy_backend
    assert mock_sentence_transformer.call_count == 1


def test_factory_equivalent_maps_with_different_insertion_order_share_identity(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        first = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="order_dense_model", tokenizer_kwargs={"b": 2, "a": 1}
        )
        second = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="order_dense_model", processor_kwargs={"a": 1, "b": 2}
        )
    assert second is first
    assert mock_sentence_transformer.call_count == 1


def test_factory_different_effective_values_are_isolated(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        first = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="isolated_dense_model", processor_kwargs={"model_max_length": 128}
        )
        second = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="isolated_dense_model", processor_kwargs={"model_max_length": 256}
        )
        third = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="isolated_dense_model", tokenizer_kwargs={"model_max_length": 512}
        )
    assert second is not first
    assert third is not first
    assert third is not second
    assert mock_sentence_transformer.call_count == 3


def test_factory_cache_hit_with_legacy_still_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET):
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            first = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model="hit_dense_model", processor_kwargs={"model_max_length": 128}
            )
            assert _util_warnings(caplog) == []
            second = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model="hit_dense_model", tokenizer_kwargs={"model_max_length": 128}
            )
    assert second is first
    records = _util_warnings(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == _DEPRECATED_MESSAGE


def test_factory_backend_layering_warns_only_once(monkeypatch, caplog):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET):
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model="layered_dense_model", tokenizer_kwargs={"model_max_length": 128}
            )
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE]


def test_factory_forwards_unrelated_kwargs_unchanged(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersEmbeddingBackendFactory, "_instances", {})
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
            model="passthrough_dense_model",
            device="cpu",
            auth_token=Secret.from_token("fake-api-token"),
            trust_remote_code=True,
            revision="main",
            local_files_only=True,
            truncate_dim=256,
            model_kwargs={"torch_dtype": "float16"},
            processor_kwargs={"model_max_length": 128},
            config_kwargs={"output_attentions": True},
            backend="torch",
        )
    mock_sentence_transformer.assert_called_once_with(
        model_name_or_path="passthrough_dense_model",
        device="cpu",
        token="fake-api-token",
        trust_remote_code=True,
        revision="main",
        local_files_only=True,
        truncate_dim=256,
        model_kwargs={"torch_dtype": "float16"},
        processor_kwargs={"model_max_length": 128},
        config_kwargs={"output_attentions": True},
        backend="torch",
    )


def test_backend_direct_legacy_call_warns_and_forwards_canonical(caplog):
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersEmbeddingBackend(
                model="direct_dense_model", tokenizer_kwargs={"model_max_length": 128}
            )
    _, kwargs = mock_sentence_transformer.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 128}
    assert "tokenizer_kwargs" not in kwargs
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE]


def test_backend_direct_canonical_call_is_quiet(caplog):
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersEmbeddingBackend(
                model="direct_canonical_dense_model", processor_kwargs={"model_max_length": 256}
            )
    _, kwargs = mock_sentence_transformer.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 256}
    assert "tokenizer_kwargs" not in kwargs
    assert _util_warnings(caplog) == []


def test_backend_direct_both_prefers_canonical_with_conflict_log(caplog):
    with patch(_PATCH_TARGET) as mock_sentence_transformer:
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersEmbeddingBackend(
                model="direct_both_dense_model",
                tokenizer_kwargs={"model_max_length": 128},
                processor_kwargs={"model_max_length": 256},
            )
    _, kwargs = mock_sentence_transformer.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 256}
    assert "tokenizer_kwargs" not in kwargs
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE + _CONFLICT_SUFFIX]
