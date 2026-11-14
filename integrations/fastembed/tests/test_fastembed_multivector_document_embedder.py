# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import Document, default_from_dict

from haystack_integrations.components.embedders.fastembed import FastembedMultivectorDocumentEmbedder

MODULE = "haystack_integrations.components.embedders.fastembed.fastembed_multivector_document_embedder"


def test_init_and_serialization():
    embedder = FastembedMultivectorDocumentEmbedder(
        model="custom",
        cache_dir="cache",
        threads=2,
        prefix="passage: ",
        suffix="!",
        batch_size=4,
        progress_bar=False,
        parallel=3,
        local_files_only=True,
        meta_fields_to_embed=["title"],
        embedding_separator=" | ",
        model_kwargs={"providers": ["CPUExecutionProvider"]},
    )
    assert embedder._model is None
    data = embedder.to_dict()
    assert data == {
        "type": f"{MODULE}.FastembedMultivectorDocumentEmbedder",
        "init_parameters": {
            "model": "custom",
            "cache_dir": "cache",
            "threads": 2,
            "prefix": "passage: ",
            "suffix": "!",
            "batch_size": 4,
            "progress_bar": False,
            "parallel": 3,
            "local_files_only": True,
            "meta_fields_to_embed": ["title"],
            "embedding_separator": " | ",
            "model_kwargs": {"providers": ["CPUExecutionProvider"]},
        },
    }
    restored = default_from_dict(FastembedMultivectorDocumentEmbedder, data)
    assert restored.to_dict() == data
    assert restored._model is None
    assert embedder.run._output_types_cache["documents"].type == list[Document]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        FastembedMultivectorDocumentEmbedder(batch_size=batch_size)


def test_warm_up_is_lazy_and_idempotent():
    with patch(f"{MODULE}.LateInteractionTextEmbedding") as model:
        embedder = FastembedMultivectorDocumentEmbedder(model="custom", local_files_only=True)
        model.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
    model.assert_called_once_with(model_name="custom", cache_dir=None, threads=None, local_files_only=True)


def test_run_embeds_metadata_and_returns_copies_in_input_order():
    embedder = FastembedMultivectorDocumentEmbedder(
        prefix="[",
        suffix="]",
        batch_size=2,
        parallel=1,
        progress_bar=False,
        meta_fields_to_embed=["title", "missing", "priority"],
        embedding_separator=" | ",
    )
    embedder._model = MagicMock()
    embedder._model.embed.return_value = iter(
        [
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[0.5, 0.6]]),
        ]
    )
    originals = [
        Document(
            id="same",
            content="Paris",
            meta={"title": "City", "missing": None, "priority": 1},
            embedding=[9.0],
            multivector_embedding=[[8.0]],
        ),
        Document(id="same", content=None),
    ]

    result = embedder.run(originals)
    embedder._model.embed.assert_called_once_with(
        ["[City | 1 | Paris]", "[]"],
        batch_size=2,
        parallel=1,
    )
    assert set(result) == {"documents"}
    assert [doc.multivector_embedding for doc in result["documents"]] == [
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6]],
    ]
    assert all(copy is not original for copy, original in zip(result["documents"], originals, strict=True))
    assert [doc.id for doc in result["documents"]] == ["same", "same"]
    assert result["documents"][0].embedding == [9.0]
    assert originals[0].multivector_embedding == [[8.0]]
    assert originals[1].multivector_embedding is None


def test_empty_documents_do_not_load_model():
    embedder = FastembedMultivectorDocumentEmbedder()
    with patch.object(embedder, "warm_up") as warm_up:
        assert embedder.run([]) == {"documents": []}
    warm_up.assert_not_called()


@pytest.mark.integration
def test_real_model():
    embedder = FastembedMultivectorDocumentEmbedder(progress_bar=False)
    originals = [Document(content="Paris is the capital of France.")]
    result = embedder.run(originals)
    vectors = result["documents"][0].multivector_embedding
    assert vectors and all(vector and all(isinstance(value, float) for value in vector) for vector in vectors)
    assert originals[0].multivector_embedding is None
