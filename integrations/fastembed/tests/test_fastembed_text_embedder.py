from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import default_from_dict
from haystack_integrations.components.embedders.fastembed.fastembed_text_embedder import (
    FastembedTextEmbedder,
)


class TestFastembedTextEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedTextEmbedder.
        """
        embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 256
        assert embedder.progress_bar is True
        assert embedder.parallel is None

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for FastembedTextEmbedder.
        """
        embedder = FastembedTextEmbedder(
            model="BAAI/bge-small-en-v1.5",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            parallel=1,
        )
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1

    def test_to_dict(self):
        """
        Test serialization of FastembedTextEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder",  # noqa
            "init_parameters": {
                "model": "BAAI/bge-small-en-v1.5",
                "prefix": "",
                "suffix": "",
                "batch_size": 256,
                "progress_bar": True,
                "parallel": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedTextEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = FastembedTextEmbedder(
            model="BAAI/bge-small-en-v1.5",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            parallel=1,
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder",  # noqa
            "init_parameters": {
                "model": "BAAI/bge-small-en-v1.5",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of FastembedTextEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder",  # noqa
            "init_parameters": {
                "model": "BAAI/bge-small-en-v1.5",
                "prefix": "",
                "suffix": "",
                "batch_size": 256,
                "progress_bar": True,
                "parallel": None,
            },
        }
        embedder = default_from_dict(FastembedTextEmbedder, embedder_dict)
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 256
        assert embedder.progress_bar is True
        assert embedder.parallel is None

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of FastembedTextEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder",  # noqa
            "init_parameters": {
                "model": "BAAI/bge-small-en-v1.5",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
            },
        }
        embedder = default_from_dict(FastembedTextEmbedder, embedder_dict)
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_text_embedder._FastembedEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_text_embedder._FastembedEmbeddingBackendFactory"
    )
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_embed(self):
        """
        Test for checking output dimensions and embedding dimensions.
        """
        embedder = FastembedTextEmbedder(model="BAAI/bge-base-en-v1.5")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()  # noqa: ARG005

        text = "Good text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(emb, float) for emb in embedding)

    def test_run_wrong_incorrect_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = FastembedTextEmbedder(model="BAAI/bge-base-en-v1.5")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="FastembedTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run(self):
        embedder = FastembedTextEmbedder(
            model="BAAI/bge-small-en-v1.5",
        )
        embedder.warm_up()

        text = "Parton energy loss in QCD matter"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(emb, float) for emb in embedding)
