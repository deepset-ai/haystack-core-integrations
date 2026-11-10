# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend import (
    _SentenceTransformersMultivectorEmbeddingBackendFactory,
)


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend.MultiVectorEncoder"
)
def test_factory_behavior(mock_multivector_encoder):
    embedding_backend = _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    same_embedding_backend = _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    another_embedding_backend = _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
        model="another_model", device="cpu"
    )
    yet_another_embedding_backend = _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu", trust_remote_code=True
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend
    assert yet_another_embedding_backend is not embedding_backend


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend.MultiVectorEncoder"
)
def test_model_initialization(mock_multivector_encoder):
    _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
        model="model",
        device="cpu",
        auth_token=Secret.from_token("fake-api-token"),
        trust_remote_code=True,
        local_files_only=True,
        backend="torch",
    )
    mock_multivector_encoder.assert_called_once_with(
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
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend.MultiVectorEncoder"
)
def test_model_initialization_onnx_session_options(mock_multivector_encoder):
    _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
        model="onnx_session_model", model_kwargs={"session_options": {"intra_op_num_threads": 2}}, backend="onnx"
    )

    session_options = mock_multivector_encoder.call_args.kwargs["model_kwargs"]["session_options"]
    assert isinstance(session_options, ort.SessionOptions)
    assert session_options.intra_op_num_threads == 2


@patch(
    "haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend.MultiVectorEncoder"
)
@pytest.mark.parametrize("is_query", [True, False])
def test_embedding_function_with_kwargs(mock_multivector_encoder, is_query):
    _SentenceTransformersMultivectorEmbeddingBackendFactory._instances = {}
    embedding_backend = _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(model="model")
    data = ["sentence1", "sentence2"]
    encode = embedding_backend.model.encode_query if is_query else embedding_backend.model.encode_document
    encode.return_value = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6]])]

    embeddings = embedding_backend.embed(
        data=data, is_query=is_query, batch_size=2, show_progress_bar=True, prompt="custom prompt"
    )

    encode.assert_called_once_with(
        data, batch_size=2, show_progress_bar=True, convert_to_numpy=True, prompt="custom prompt"
    )
    other_encode = embedding_backend.model.encode_document if is_query else embedding_backend.model.encode_query
    other_encode.assert_not_called()
    assert embeddings == [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]]
