from unittest.mock import MagicMock

import pytest
from haystack import Document, default_from_dict

from haystack_integrations.components.rankers.fastembed.ranker import (
    FastembedRanker,
)


class TestFastembedRanker:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedRanker.
        """
        ranker = FastembedRanker(model_name="BAAI/bge-reranker-base")
        assert ranker.model_name == "BAAI/bge-reranker-base"
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
        Test custom initialization parameters for FastembedRanker.
        """
        ranker = FastembedRanker(
            model_name="BAAI/bge-reranker-base",
            top_k=64,
            cache_dir="fake_dir",
            threads=2,
            batch_size=50,
            parallel=1,
            local_files_only=True,
            meta_fields_to_embed=["test_field"],
            meta_data_separator=" | ",
        )
        assert ranker.model_name == "BAAI/bge-reranker-base"
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
        Test for checking incorrect input format on init
        """
        with pytest.raises(
            ValueError,
            match="top_k must be > 0, but got 0",
        ):
            FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-12-v2", top_k=0)

        with pytest.raises(
            ValueError,
            match="top_k must be > 0, but got -3",
        ):
            FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-12-v2", top_k=-3)

    def test_to_dict(self):
        """
        Test serialization of FastembedRanker to a dictionary, using default initialization parameters.
        """
        ranker = FastembedRanker(model_name="BAAI/bge-reranker-base")
        ranker_dict = ranker.to_dict()
        assert ranker_dict == {
            "type": "haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker",
            "init_parameters": {
                "model_name": "BAAI/bge-reranker-base",
                "top_k": 10,
                "cache_dir": None,
                "threads": None,
                "batch_size": 64,
                "parallel": None,
                "local_files_only": False,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedRanker to a dictionary, using custom initialization parameters.
        """
        ranker = FastembedRanker(
            model_name="BAAI/bge-reranker-base",
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
            "type": "haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker",
            "init_parameters": {
                "model_name": "BAAI/bge-reranker-base",
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

    def test_from_dict(self):
        """
        Test deserialization of FastembedRanker from a dictionary, using default initialization parameters.
        """
        ranker_dict = {
            "type": "haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker",
            "init_parameters": {
                "model_name": "BAAI/bge-reranker-base",
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
        ranker = default_from_dict(FastembedRanker, ranker_dict)
        assert ranker.model_name == "BAAI/bge-reranker-base"
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
        Test deserialization of FastembedRanker from a dictionary, using custom initialization parameters.
        """
        ranker_dict = {
            "type": "haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker",
            "init_parameters": {
                "model_name": "BAAI/bge-reranker-base",
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
        ranker = default_from_dict(FastembedRanker, ranker_dict)
        assert ranker.model_name == "BAAI/bge-reranker-base"
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
        ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-12-v2")
        ranker._model = "mock_model"

        query = "query"
        string_input = "text"
        list_integers_input = [1, 2, 3]
        list_document = [Document("Document 1")]

        with pytest.raises(
            TypeError,
            match=r"FastembedRanker expects a list of Documents as input\.",
        ):
            ranker.run(query=query, documents=string_input)

        with pytest.raises(
            TypeError,
            match=r"FastembedRanker expects a list of Documents as input\.",
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

    def test_run_no_warmup(self):
        """
        Test for checking error when calling without a warmup.
        """
        ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-12-v2")

        query = "query"
        list_document = [Document("Document 1")]

        with pytest.raises(
            RuntimeError,
        ):
            ranker.run(query=query, documents=list_document)

    def test_run_empty_document_list(self):
        """
        Test for no error when sending no documents.
        """
        ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-12-v2")
        ranker._model = "mock_model"

        query = "query"
        list_document = []

        result = ranker.run(query=query, documents=list_document)
        assert len(result["documents"]) == 0

    def test_embed_metadata(self):
        """
        Tests the embedding of metadata fields in document content for ranking.
        """
        ranker = FastembedRanker(
            model_name="model_name",
            meta_fields_to_embed=["meta_field"],
        )
        ranker._model = MagicMock()

        documents = [Document(content=f"document-number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]
        query = "test"
        ranker.run(query=query, documents=documents)

        ranker._model.rerank.assert_called_once_with(
            query=query,
            documents=[
                "meta_value 0\ndocument-number 0",
                "meta_value 1\ndocument-number 1",
                "meta_value 2\ndocument-number 2",
                "meta_value 3\ndocument-number 3",
                "meta_value 4\ndocument-number 4",
            ],
            batch_size=64,
            parallel=None,
        )

    @pytest.mark.integration
    def test_run(self):
        ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-6-v2", top_k=2)
        ranker.warm_up()

        query = "Who is maintaining Qdrant?"
        documents = [
            Document(
                content="This is built to be faster and lighter than other embedding \
libraries e.g. Transformers, Sentence-Transformers, etc."
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
