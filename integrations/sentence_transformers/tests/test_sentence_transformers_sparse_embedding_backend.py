# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.sentence_transformers.embedding_backend.sparse_backend import (
    _SentenceTransformersSparseEmbeddingBackendFactory,
    _SentenceTransformersSparseEncoderEmbeddingBackend,
)


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.sparse_backend.SparseEncoder"
)
def test_sparse_factory_behavior(mock_sparse_encoder):
    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    same_embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    another_embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="another_model", device="cpu"
    )
    yet_another_embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu", trust_remote_code=True
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend
    assert yet_another_embedding_backend is not embedding_backend


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.sparse_backend.SparseEncoder"
)
def test_sparse_model_initialization(mock_sparse_encoder):
    _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="model",
        device="cpu",
        auth_token=Secret.from_token("fake-api-token"),
        trust_remote_code=True,
        local_files_only=True,
        backend="torch",
    )
    mock_sparse_encoder.assert_called_once_with(
        model_name_or_path="model",
        device="cpu",
        token="fake-api-token",
        trust_remote_code=True,
        revision=None,
        local_files_only=True,
        model_kwargs=None,
        processor_kwargs=None,
        config_kwargs=None,
        backend="torch",
    )


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.sparse_backend.SparseEncoder"
)
def test_sparse_embedding_function_with_kwargs(mock_sparse_encoder):
    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(model="model")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data, attn_implementation="sdpa")

    embedding_backend.model.encode.assert_called_once_with(
        data, convert_to_tensor=False, convert_to_sparse_tensor=True, attn_implementation="sdpa"
    )


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.sparse_backend.SparseEncoder"
)
def test_sparse_embedding_function(mock_sparse_encoder):
    """
    Test that the backend's embed method returns the correct sparse embeddings.
    """

    # Ensure the factory cache is cleared before each test.
    _SentenceTransformersSparseEmbeddingBackendFactory._instances = {}

    tensors = [
        torch.sparse_coo_tensor(torch.tensor([[1, 4]]), torch.tensor([0.5, 0.8])),
        torch.sparse_coo_tensor(torch.tensor([[2]]), torch.tensor([0.3])),
    ]
    mock_sparse_encoder.return_value.encode.return_value = tensors

    # Get the embedding backend
    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(model="model")

    # Embed dummy data
    data = ["sentence1", "sentence2"]
    sparse_embeddings = embedding_backend.embed(data=data)

    # Expected output
    expected_embeddings = [
        SparseEmbedding(indices=[1, 4], values=[0.5, 0.8]),
        SparseEmbedding(indices=[2], values=[0.3]),
    ]

    assert len(sparse_embeddings) == len(expected_embeddings)
    for got, exp in zip(sparse_embeddings, expected_embeddings, strict=True):
        assert got.indices == exp.indices
        assert got.values == pytest.approx(exp.values)


_SPARSE_PATCH_TARGET = (
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.sparse_backend.SparseEncoder"
)
_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."


def _util_warnings(caplog):
    return [record for record in caplog.records if record.name == _UTIL_LOGGER and record.levelname == "WARNING"]


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
def test_sparse_factory_processor_kwargs_matrix(monkeypatch, legacy, canonical, expected):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    legacy_snapshot = dict(legacy) if legacy is not None else None
    canonical_snapshot = dict(canonical) if canonical is not None else None
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="matrix_sparse_model", tokenizer_kwargs=legacy, processor_kwargs=canonical
        )
    _, kwargs = mock_sparse_encoder.call_args
    assert kwargs["processor_kwargs"] == expected
    assert "tokenizer_kwargs" not in kwargs
    assert legacy == legacy_snapshot
    assert canonical == canonical_snapshot


def test_sparse_factory_explicit_canonical_none_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="explicit_none_sparse_model",
            tokenizer_kwargs={"model_max_length": 128},
            processor_kwargs=None,
        )
    _, kwargs = mock_sparse_encoder.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 128}
    assert "tokenizer_kwargs" not in kwargs


def test_sparse_factory_alias_spellings_share_cache_identity(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        legacy_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="alias_sparse_model", tokenizer_kwargs={"model_max_length": 128}
        )
        canonical_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="alias_sparse_model", processor_kwargs={"model_max_length": 128}
        )
        both_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="alias_sparse_model",
            tokenizer_kwargs={"other": 1},
            processor_kwargs={"model_max_length": 128},
        )
    assert canonical_backend is legacy_backend
    assert both_backend is legacy_backend
    assert mock_sparse_encoder.call_count == 1


def test_sparse_factory_equivalent_maps_with_different_insertion_order_share_identity(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        first = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="order_sparse_model", tokenizer_kwargs={"b": 2, "a": 1}
        )
        second = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="order_sparse_model", processor_kwargs={"a": 1, "b": 2}
        )
    assert second is first
    assert mock_sparse_encoder.call_count == 1


def test_sparse_factory_different_effective_values_are_isolated(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        first = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="isolated_sparse_model", processor_kwargs={"model_max_length": 128}
        )
        second = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="isolated_sparse_model", processor_kwargs={"model_max_length": 256}
        )
        third = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="isolated_sparse_model", tokenizer_kwargs={"model_max_length": 512}
        )
    assert second is not first
    assert third is not first
    assert third is not second
    assert mock_sparse_encoder.call_count == 3


def test_sparse_factory_cache_hit_with_legacy_still_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET):
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            first = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
                model="hit_sparse_model", processor_kwargs={"model_max_length": 128}
            )
            assert _util_warnings(caplog) == []
            second = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
                model="hit_sparse_model", tokenizer_kwargs={"model_max_length": 128}
            )
    assert second is first
    records = _util_warnings(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == _DEPRECATED_MESSAGE


def test_sparse_factory_backend_layering_warns_only_once(monkeypatch, caplog):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET):
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
                model="layered_sparse_model", tokenizer_kwargs={"model_max_length": 128}
            )
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE]


def test_sparse_factory_forwards_unrelated_kwargs_unchanged(monkeypatch):
    monkeypatch.setattr(_SentenceTransformersSparseEmbeddingBackendFactory, "_instances", {})
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
            model="passthrough_sparse_model",
            device="cpu",
            auth_token=Secret.from_token("fake-api-token"),
            trust_remote_code=True,
            revision="main",
            local_files_only=True,
            model_kwargs={"torch_dtype": "float16"},
            processor_kwargs={"model_max_length": 128},
            config_kwargs={"output_attentions": True},
            backend="torch",
        )
    mock_sparse_encoder.assert_called_once_with(
        model_name_or_path="passthrough_sparse_model",
        device="cpu",
        token="fake-api-token",
        trust_remote_code=True,
        revision="main",
        local_files_only=True,
        model_kwargs={"torch_dtype": "float16"},
        processor_kwargs={"model_max_length": 128},
        config_kwargs={"output_attentions": True},
        backend="torch",
    )


def test_sparse_backend_direct_legacy_call_warns_and_forwards_canonical(caplog):
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersSparseEncoderEmbeddingBackend(
                model="direct_sparse_model", tokenizer_kwargs={"model_max_length": 128}
            )
    _, kwargs = mock_sparse_encoder.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 128}
    assert "tokenizer_kwargs" not in kwargs
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE]


def test_sparse_backend_direct_canonical_call_is_quiet(caplog):
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersSparseEncoderEmbeddingBackend(
                model="direct_canonical_sparse_model", processor_kwargs={"model_max_length": 256}
            )
    _, kwargs = mock_sparse_encoder.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 256}
    assert "tokenizer_kwargs" not in kwargs
    assert _util_warnings(caplog) == []


def test_sparse_backend_direct_both_prefers_canonical_with_conflict_log(caplog):
    with patch(_SPARSE_PATCH_TARGET) as mock_sparse_encoder:
        with caplog.at_level("WARNING", logger=_UTIL_LOGGER):
            _SentenceTransformersSparseEncoderEmbeddingBackend(
                model="direct_both_sparse_model",
                tokenizer_kwargs={"model_max_length": 128},
                processor_kwargs={"model_max_length": 256},
            )
    _, kwargs = mock_sparse_encoder.call_args
    assert kwargs["processor_kwargs"] == {"model_max_length": 256}
    assert "tokenizer_kwargs" not in kwargs
    records = _util_warnings(caplog)
    assert [record.getMessage() for record in records] == [_DEPRECATED_MESSAGE + _CONFLICT_SUFFIX]
