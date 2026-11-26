# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import default_from_dict

from haystack_integrations.components.embedders.fastembed import FastembedMultivectorTextEmbedder

MODULE = "haystack_integrations.components.embedders.fastembed.fastembed_multivector_text_embedder"


def test_init_and_serialization():
    embedder = FastembedMultivectorTextEmbedder(
        model="custom",
        cache_dir="cache",
        threads=2,
        prefix="query: ",
        suffix="!",
        local_files_only=True,
        model_kwargs={"providers": ["CPUExecutionProvider"]},
    )
    assert embedder._model is None
    data = embedder.to_dict()
    assert data == {
        "type": f"{MODULE}.FastembedMultivectorTextEmbedder",
        "init_parameters": {
            "model": "custom",
            "cache_dir": "cache",
            "threads": 2,
            "prefix": "query: ",
            "suffix": "!",
            "local_files_only": True,
            "model_kwargs": {"providers": ["CPUExecutionProvider"]},
        },
    }
    restored = default_from_dict(FastembedMultivectorTextEmbedder, data)
    assert restored.to_dict() == data
    assert restored._model is None
    assert embedder.run._output_types_cache["multivector_embedding"].type == list[list[float]]


def test_warm_up_is_lazy_and_idempotent():
    with patch(f"{MODULE}.LateInteractionTextEmbedding") as model:
        embedder = FastembedMultivectorTextEmbedder(
            model="custom", local_files_only=True, model_kwargs={"providers": ["CPUExecutionProvider"]}
        )
        model.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
    model.assert_called_once_with(
        model_name="custom",
        cache_dir=None,
        threads=None,
        local_files_only=True,
        providers=["CPUExecutionProvider"],
    )


def test_run_uses_query_embedding_and_preserves_token_vectors():
    embedder = FastembedMultivectorTextEmbedder(prefix="[", suffix="]")
    embedder._model = MagicMock()
    embedder._model.query_embed.return_value = iter([np.array([[0.1, 0.2], [0.3, 0.4]])])

    assert embedder.run("Paris") == {"multivector_embedding": [[0.1, 0.2], [0.3, 0.4]]}
    embedder._model.query_embed.assert_called_once_with("[Paris]")


def test_run_loads_model_when_needed():
    embedder = FastembedMultivectorTextEmbedder()
    with patch(f"{MODULE}.LateInteractionTextEmbedding") as model:
        model.return_value.query_embed.return_value = iter([np.array([[0.1, 0.2]])])
        assert embedder.run("Paris") == {"multivector_embedding": [[0.1, 0.2]]}
    model.assert_called_once()
    model.return_value.query_embed.assert_called_once_with("Paris")


@pytest.mark.integration
def test_real_model():
    embedder = FastembedMultivectorTextEmbedder()
    result = embedder.run("What is the capital of France?")
    vectors = result["multivector_embedding"]
    assert vectors and all(vector and all(isinstance(value, float) for value in vector) for vector in vectors)
