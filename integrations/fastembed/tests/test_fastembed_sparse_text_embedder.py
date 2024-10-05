from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import default_from_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding

from haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder import (
    FastembedSparseTextEmbedder,
)


class TestFastembedSparseTextEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedSparseTextEmbedder.
        """
        embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")
        assert embedder.model_name == "prithvida/Splade_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.progress_bar is True
        assert embedder.parallel is None

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for FastembedSparseTextEmbedder.
        """
        embedder = FastembedSparseTextEmbedder(
            model="prithvida/Splade_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            progress_bar=False,
            parallel=1,
        )
        assert embedder.model_name == "prithvida/Splade_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.progress_bar is False
        assert embedder.parallel == 1

    def test_to_dict(self):
        """
        Test serialization of FastembedSparseTextEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/Splade_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "progress_bar": True,
                "parallel": None,
                "local_files_only": False,
                "model_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedSparseTextEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = FastembedSparseTextEmbedder(
            model="prithvida/Splade_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            progress_bar=False,
            parallel=1,
            local_files_only=True,
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/Splade_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "progress_bar": False,
                "parallel": 1,
                "local_files_only": True,
                "model_kwargs": None,
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of FastembedSparseTextEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/Splade_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "progress_bar": True,
                "parallel": None,
            },
        }
        embedder = default_from_dict(FastembedSparseTextEmbedder, embedder_dict)
        assert embedder.model_name == "prithvida/Splade_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.progress_bar is True
        assert embedder.parallel is None

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of FastembedSparseTextEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/Splade_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "progress_bar": False,
                "parallel": 1,
            },
        }
        embedder = default_from_dict(FastembedSparseTextEmbedder, embedder_dict)
        assert embedder.model_name == "prithvida/Splade_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.progress_bar is False
        assert embedder.parallel == 1

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name="prithvida/Splade_PP_en_v1",
            cache_dir=None,
            threads=None,
            local_files_only=False,
            model_kwargs=None,
        )

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def _generate_mocked_sparse_embedding(self, n):
        list_of_sparse_vectors = []
        for _ in range(n):
            random_indice_length = np.random.randint(3, 15)
            data = {
                "indices": list(range(random_indice_length)),
                "values": [np.random.random_sample() for _ in range(random_indice_length)],
            }
            list_of_sparse_vectors.append(data)

        return list_of_sparse_vectors

    def test_embed(self):
        """
        Test for checking output dimensions and embedding dimensions.
        """
        embedder = FastembedSparseTextEmbedder(model="BAAI/bge-base-en-v1.5")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: self._generate_mocked_sparse_embedding(  # noqa: ARG005
            len(x)
        )

        text = "Good text to embed"

        result = embedder.run(text=text)
        embedding = result["sparse_embedding"]
        assert isinstance(embedding, dict)
        assert isinstance(embedding["indices"], list)
        assert isinstance(embedding["indices"][0], int)
        assert isinstance(embedding["values"], list)
        assert isinstance(embedding["values"][0], float)

    def test_run_wrong_incorrect_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = FastembedSparseTextEmbedder(model="BAAI/bge-base-en-v1.5")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="FastembedSparseTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    def test_init_with_model_kwargs_parameters(self):
        """
        Test initialization of FastembedSparseTextEmbedder with model_kwargs parameters.
        """
        bm25_config = {
            "k": 1.2,
            "b": 0.75,
            "avg_len": 300.0,
            "language": "english",
            "token_max_length": 50,
        }

        embedder = FastembedSparseTextEmbedder(
            model="Qdrant/bm25",
            model_kwargs=bm25_config,
        )

        assert embedder.model_kwargs == bm25_config

    @pytest.mark.integration
    def test_run_with_model_kwargs(self):
        """
        Integration test to check the embedding with model_kwargs parameters.
        """
        bm25_config = {
            "k": 1.2,
            "b": 0.75,
            "avg_len": 256.0,
        }

        embedder = FastembedSparseTextEmbedder(
            model="Qdrant/bm25",
            model_kwargs=bm25_config,
        )
        embedder.warm_up()

        text = "Example content using BM25"

        result = embedder.run(text=text)
        embedding = result["sparse_embedding"]
        embedding_dict = embedding.to_dict()

        assert isinstance(embedding, SparseEmbedding)
        assert isinstance(embedding_dict["indices"], list)
        assert isinstance(embedding_dict["values"], list)
        assert isinstance(embedding_dict["indices"][0], int)
        assert isinstance(embedding_dict["values"][0], float)

    @pytest.mark.integration
    def test_run(self):
        embedder = FastembedSparseTextEmbedder(
            model="prithvida/Splade_PP_en_v1",
        )
        embedder.warm_up()

        text = "Parton energy loss in QCD matter"

        result = embedder.run(text=text)
        embedding = result["sparse_embedding"]
        embedding_dict = embedding.to_dict()
        assert isinstance(embedding, SparseEmbedding)
        assert isinstance(embedding_dict["indices"], list)
        assert isinstance(embedding_dict["values"], list)
        assert isinstance(embedding_dict["indices"][0], int)
        assert isinstance(embedding_dict["values"][0], float)
