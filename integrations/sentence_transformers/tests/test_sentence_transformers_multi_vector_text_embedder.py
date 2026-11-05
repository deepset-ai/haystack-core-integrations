# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
from haystack import Pipeline
from haystack.utils import Secret

from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersMultiVectorTextEmbedder


def test_query_encoding_and_pipeline_round_trip():
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        SentenceTransformersMultiVectorTextEmbedder(
            model="test-model", device="cpu", token=Secret.from_env_var("TEST_HF_TOKEN", strict=False), prefix="Q: "
        ),
    )
    pipeline = Pipeline.loads(pipeline.dumps())
    with patch(
        "haystack_integrations.components.embedders.sentence_transformers."
        "sentence_transformers_multi_vector_text_embedder.MultiVectorEncoder"
    ) as encoder:
        encoder.return_value.encode_query.return_value = [np.array([[1.0, 0.0], [0.0, 1.0]])]
        result = pipeline.run({"embedder": {"text": "hello"}})
        assert result["embedder"]["multi_vector_embedding"] == [[1.0, 0.0], [0.0, 1.0]]
        encoder.assert_called_once_with(
            model_name_or_path="test-model", device="cpu", token=None, trust_remote_code=False
        )
        encoder.return_value.encode_query.assert_called_once_with(
            ["Q: hello"], convert_to_numpy=True, show_progress_bar=False
        )
        pipeline.run({"embedder": {"text": "again"}})
        assert encoder.call_count == 1
