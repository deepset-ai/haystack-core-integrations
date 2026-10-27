# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
from haystack import Document, Pipeline
from haystack.components.rankers import MaxSimRanker

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersMultiVectorDocumentEmbedder,
)


def test_document_encoding_and_ranking():
    documents = [
        Document(id="partial", content="one", meta={"title": "First"}, embedding=[0.1, 0.2]),
        Document(id="best", content="two", meta={"title": "Second"}),
    ]
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        SentenceTransformersMultiVectorDocumentEmbedder(
            model="test-model", token=None, batch_size=2, meta_fields_to_embed=["title"], embedding_separator=" | "
        ),
    )
    pipeline.add_component("ranker", MaxSimRanker())
    pipeline.connect("embedder.documents", "ranker.documents")
    pipeline = Pipeline.loads(pipeline.dumps())

    with patch(
        "haystack_integrations.components.embedders.sentence_transformers."
        "sentence_transformers_multi_vector_document_embedder.MultiVectorEncoder"
    ) as encoder:
        encoder.return_value.encode_document.return_value = [
            np.array([[0.5, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ]
        result = pipeline.run(
            {"embedder": {"documents": documents}, "ranker": {"query_embedding": [[1.0, 0.0], [0.0, 1.0]]}}
        )["ranker"]["documents"]
        encoder.return_value.encode_document.assert_called_once_with(
            ["First | one", "Second | two"], batch_size=2, convert_to_numpy=True, show_progress_bar=False
        )

    assert [(doc.id, doc.score) for doc in result] == [("best", 2.0), ("partial", 0.5)]
    assert result[0].multi_vector_embedding == [[1.0, 0.0], [0.0, 1.0]]
    assert result[1].embedding == [0.1, 0.2]
    assert result[1].meta == {"title": "First"}
    assert all(doc.multi_vector_embedding is None for doc in documents)


def test_empty_documents_do_not_load_model():
    with patch(
        "haystack_integrations.components.embedders.sentence_transformers."
        "sentence_transformers_multi_vector_document_embedder.MultiVectorEncoder"
    ) as encoder:
        assert SentenceTransformersMultiVectorDocumentEmbedder().run(documents=[]) == {"documents": []}
        encoder.assert_not_called()
