from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import default_from_dict
from haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder import (
    FastembedTextSPLADEEmbedder,
)


class TestFastembedTextSPLADEEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedTextSPLADEEmbedder.
        """
        embedder = FastembedTextSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 256
        assert embedder.progress_bar is True
        assert embedder.parallel is None

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for FastembedTextSPLADEEmbedder.
        """
        embedder = FastembedTextSPLADEEmbedder(
            model="prithvida/SPLADE_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            parallel=1,
        )
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1

    def test_to_dict(self):
        """
        Test serialization of FastembedTextSPLADEEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = FastembedTextSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder.FastembedTextSPLADEEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 256,
                "progress_bar": True,
                "parallel": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedTextSPLADEEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = FastembedTextSPLADEEmbedder(
            model="prithvida/SPLADE_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            parallel=1,
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder.FastembedTextSPLADEEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of FastembedTextSPLADEEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder.FastembedTextSPLADEEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 256,
                "progress_bar": True,
                "parallel": None,
            },
        }
        embedder = default_from_dict(FastembedTextSPLADEEmbedder, embedder_dict)
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 256
        assert embedder.progress_bar is True
        assert embedder.parallel is None

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of FastembedTextSPLADEEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder.FastembedTextSPLADEEmbedder",  # noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
            },
        }
        embedder = default_from_dict(FastembedTextSPLADEEmbedder, embedder_dict)
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = FastembedTextSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name="prithvida/SPLADE_PP_en_v1", cache_dir=None, threads=None
        )

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_text_SPLADE_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = FastembedTextSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def _generate_mocked_sparse_embedding(self, n):
        list_of_sparse_vectors = []
        for _ in range(n):
            random_indice_length = np.random.randint(0, 20)
            data = {
                "indices": [i for i in range(random_indice_length)],
                "values": [np.random.random_sample() for _ in range(random_indice_length)]
            }
            list_of_sparse_vectors.append(data)

        return list_of_sparse_vectors

    def test_embed(self):
        """
        Test for checking output dimensions and embedding dimensions.
        """
        embedder = FastembedTextSPLADEEmbedder(model="BAAI/bge-base-en-v1.5")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: self._generate_mocked_sparse_embedding(
            len(x))  # noqa: ARG005

        text = "Good text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]
        # TODO adapt to sparse
        assert isinstance(embedding, dict)
        assert isinstance(embedding["indices"], list)
        assert isinstance(embedding["indices"][0], int)
        assert isinstance(embedding["values"], list)
        assert isinstance(embedding["values"][0], float)

    def test_run_wrong_incorrect_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = FastembedTextSPLADEEmbedder(model="BAAI/bge-base-en-v1.5")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="FastembedTextSPLADEEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run(self):
        embedder = FastembedTextSPLADEEmbedder(
            model="prithvida/SPLADE_PP_en_v1",
        )
        embedder.warm_up()

        text = "Parton energy loss in QCD matter"

        result = embedder.run(text=text)
        embedding = result["embedding"]
        # TODO adapt to sparse
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(emb, float) for emb in embedding)
