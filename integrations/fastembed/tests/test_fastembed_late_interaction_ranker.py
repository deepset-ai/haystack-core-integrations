# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import Document, default_from_dict

from haystack_integrations.components.rankers.fastembed.late_interaction_ranker import (
    FastembedLateInteractionRanker,
)


class TestFastembedLateInteractionRanker:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedLateInteractionRanker.
        """
        ranker = FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0")
        assert ranker.model_name == "colbert-ir/colbertv2.0"
        assert ranker.top_k == 10
        assert ranker.cache_dir is None
        assert ranker.threads is None
        assert ranker.batch_size == 64
        assert ranker.parallel is None
        assert not ranker.local_files_only
        assert ranker.meta_fields_to_embed == []
        assert ranker.meta_data_separator == "\n"

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for FastembedLateInteractionRanker.
        """
        ranker = FastembedLateInteractionRanker(
            model_name="colbert-ir/colbertv2.0",
            top_k=64,
            cache_dir="fake_dir",
            threads=2,
            batch_size=50,
            parallel=1,
            local_files_only=True,
            meta_fields_to_embed=["test_field"],
            meta_data_separator=" | ",
        )
        assert ranker.model_name == "colbert-ir/colbertv2.0"
        assert ranker.top_k == 64
        assert ranker.cache_dir == "fake_dir"
        assert ranker.threads == 2
        assert ranker.batch_size == 50
        assert ranker.parallel == 1
        assert ranker.local_files_only
        assert ranker.meta_fields_to_embed == ["test_field"]
        assert ranker.meta_data_separator == " | "

    def test_init_with_incorrect_input(self):
        """
        Test for checking incorrect input format on init.
        """
        with pytest.raises(
            ValueError,
            match="top_k must be > 0, but got 0",
        ):
            FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0", top_k=0)

        with pytest.raises(
            ValueError,
            match="top_k must be > 0, but got -3",
        ):
            FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0", top_k=-3)

    def test_to_dict(self):
        """
        Test serialization of FastembedLateInteractionRanker to a dictionary, using default initialization parameters.
        """
        ranker = FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0")
        ranker_dict = ranker.to_dict()
        assert ranker_dict == {
            "type": "haystack_integrations.components.rankers.fastembed"
            ".late_interaction_ranker.FastembedLateInteractionRanker",
            "init_parameters": {
                "model_name": "colbert-ir/colbertv2.0",
                "top_k": 10,
                "cache_dir": None,
                "threads": None,
                "batch_size": 64,
                "parallel": None,
                "local_files_only": False,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
                "score_threshold": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedLateInteractionRanker to a dictionary, using custom initialization parameters.
        """
        ranker = FastembedLateInteractionRanker(
            model_name="colbert-ir/colbertv2.0",
            cache_dir="fake_dir",
            threads=2,
            top_k=5,
            batch_size=50,
            parallel=1,
            local_files_only=True,
            meta_fields_to_embed=["test_field"],
            meta_data_separator=" | ",
        )
        ranker_dict = ranker.to_dict()
        assert ranker_dict == {
            "type": "haystack_integrations.components.rankers.fastembed"
            ".late_interaction_ranker.FastembedLateInteractionRanker",
            "init_parameters": {
                "model_name": "colbert-ir/colbertv2.0",
                "cache_dir": "fake_dir",
                "threads": 2,
                "top_k": 5,
                "batch_size": 50,
                "parallel": 1,
                "local_files_only": True,
                "meta_fields_to_embed": ["test_field"],
                "meta_data_separator": " | ",
                "score_threshold": None,
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of FastembedLateInteractionRanker from a dictionary, using default init parameters.
        """
        ranker_dict = {
            "type": "haystack_integrations.components.rankers.fastembed"
            ".late_interaction_ranker.FastembedLateInteractionRanker",
            "init_parameters": {
                "model_name": "colbert-ir/colbertv2.0",
                "cache_dir": None,
                "threads": None,
                "top_k": 5,
                "batch_size": 50,
                "parallel": None,
                "local_files_only": False,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
            },
        }
        ranker = default_from_dict(FastembedLateInteractionRanker, ranker_dict)
        assert ranker.model_name == "colbert-ir/colbertv2.0"
        assert ranker.cache_dir is None
        assert ranker.threads is None
        assert ranker.top_k == 5
        assert ranker.batch_size == 50
        assert ranker.parallel is None
        assert not ranker.local_files_only
        assert ranker.meta_fields_to_embed == []
        assert ranker.meta_data_separator == "\n"

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of FastembedLateInteractionRanker from a dictionary, using custom init parameters.
        """
        ranker_dict = {
            "type": "haystack_integrations.components.rankers.fastembed"
            ".late_interaction_ranker.FastembedLateInteractionRanker",
            "init_parameters": {
                "model_name": "colbert-ir/colbertv2.0",
                "cache_dir": "fake_dir",
                "threads": 2,
                "top_k": 5,
                "batch_size": 50,
                "parallel": 1,
                "local_files_only": True,
                "meta_fields_to_embed": ["test_field"],
                "meta_data_separator": " | ",
            },
        }
        ranker = default_from_dict(FastembedLateInteractionRanker, ranker_dict)
        assert ranker.model_name == "colbert-ir/colbertv2.0"
        assert ranker.cache_dir == "fake_dir"
        assert ranker.threads == 2
        assert ranker.top_k == 5
        assert ranker.batch_size == 50
        assert ranker.parallel == 1
        assert ranker.local_files_only
        assert ranker.meta_fields_to_embed == ["test_field"]
        assert ranker.meta_data_separator == " | "

    def test_run_incorrect_input_format(self):
        """
        Test for checking incorrect input format.
        """
        ranker = FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0")
        ranker._model = "mock_model"

        query = "query"
        string_input = "text"
        list_integers_input = [1, 2, 3]
        list_document = [Document("Document 1")]

        with pytest.raises(
            TypeError,
            match=r"FastembedLateInteractionRanker expects a list of Documents as input\.",
        ):
            ranker.run(query=query, documents=string_input)

        with pytest.raises(
            TypeError,
            match=r"FastembedLateInteractionRanker expects a list of Documents as input\.",
        ):
            ranker.run(query=query, documents=list_integers_input)

        with pytest.raises(
            ValueError,
            match="No query provided",
        ):
            ranker.run(query="", documents=list_document)

        with pytest.raises(
            ValueError,
            match="top_k must be > 0, but got -3",
        ):
            ranker.run(query=query, documents=list_document, top_k=-3)

    def test_run_empty_document_list(self):
        """
        Test for no error when sending no documents.
        """
        ranker = FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0")
        ranker._model = "mock_model"

        query = "query"
        list_document = []

        result = ranker.run(query=query, documents=list_document)
        assert len(result["documents"]) == 0

    def test_embed_metadata(self):
        """
        Tests the embedding of metadata fields in document content for ranking.
        """
        ranker = FastembedLateInteractionRanker(
            model_name="colbert-ir/colbertv2.0",
            meta_fields_to_embed=["meta_field"],
        )
        mock_model = MagicMock()
        # query_embed returns one embedding with shape (num_query_tokens, embedding_dim)
        mock_model.query_embed.return_value = iter([np.random.rand(32, 128)])
        # embed returns one embedding per document with shape (num_doc_tokens, embedding_dim)
        mock_model.embed.return_value = iter([np.random.rand(64, 128) for _ in range(5)])
        ranker._model = mock_model

        documents = [Document(content=f"document-number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]
        query = "test"
        ranker.run(query=query, documents=documents)

        mock_model.embed.assert_called_once_with(
            [
                "meta_value 0\ndocument-number 0",
                "meta_value 1\ndocument-number 1",
                "meta_value 2\ndocument-number 2",
                "meta_value 3\ndocument-number 3",
                "meta_value 4\ndocument-number 4",
            ],
            batch_size=64,
            parallel=None,
        )

    def test_warm_up_called_once(self):
        """
        Test that calling warm_up() twice only initializes the model once.
        """
        ranker = FastembedLateInteractionRanker()
        with patch(
            "haystack_integrations.components.rankers.fastembed.late_interaction_ranker.LateInteractionTextEmbedding"
        ) as mock_cls:
            ranker.warm_up()
            ranker.warm_up()
            mock_cls.assert_called_once()

    def test_run_calls_warm_up(self):
        """
        Unit test to check that warm_up is called when run is called for the first time.
        """
        ranker = FastembedLateInteractionRanker()

        mock_model = MagicMock()
        mock_model.query_embed.return_value = iter([np.random.rand(32, 128)])
        mock_model.embed.return_value = iter([np.random.rand(64, 128)])

        with patch.object(ranker, "warm_up", side_effect=lambda: setattr(ranker, "_model", mock_model)) as mock_warm_up:
            ranker.run(query="test query", documents=[Document(content="test document")])

        mock_warm_up.assert_called_once()

    def test_run_with_mock(self):
        """
        Test that MaxSim scoring produces correct ranking order.
        """
        ranker = FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0", top_k=2)

        mock_model = MagicMock()
        # Query embedding: 2 tokens, 4 dims
        mock_model.query_embed.return_value = iter([np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])])
        # Doc embeddings: 2 docs, each with 3 tokens, 4 dims
        # Doc 0: low similarity to query
        # Doc 1: high similarity to query
        mock_model.embed.return_value = iter(
            [
                np.array([[0.1, 0.1, 0.9, 0.0], [0.1, 0.0, 0.0, 0.9], [0.0, 0.1, 0.9, 0.0]]),
                np.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.9, 0.1, 0.0], [0.1, 0.1, 0.8, 0.0]]),
            ]
        )
        ranker._model = mock_model

        documents = [
            Document(content="low similarity doc"),
            Document(content="high similarity doc"),
        ]

        result = ranker.run(query="test", documents=documents)

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "high similarity doc"
        assert result["documents"][0].score > result["documents"][1].score

    @pytest.mark.integration
    def test_run(self):
        ranker = FastembedLateInteractionRanker(model_name="colbert-ir/colbertv2.0", top_k=2)

        query = "Who is maintaining Qdrant?"
        documents = [
            Document(
                content="This is built to be faster and lighter than other embedding "
                "libraries e.g. Transformers, Sentence-Transformers, etc."
            ),
            Document(content="This is some random input"),
            Document(content="fastembed is supported by and maintained by Qdrant."),
        ]

        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 2
        first_document = result["documents"][0]
        second_document = result["documents"][1]

        assert isinstance(first_document, Document)
        assert isinstance(second_document, Document)
        assert first_document.content == "fastembed is supported by and maintained by Qdrant."
        assert first_document.score > second_document.score
