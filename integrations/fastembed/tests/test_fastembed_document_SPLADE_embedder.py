from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import Document, default_from_dict
from haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder import (
    FastembedDocumentSPLADEEmbedder,
)


class TestFastembedDocumentSPLADEEmbedderDoc:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedDocumentSPLADEEmbedder.
        """
        embedder = FastembedDocumentSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 256
        assert embedder.progress_bar is True
        assert embedder.parallel is None
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for FastembedDocumentSPLADEEmbedder.
        """
        embedder = FastembedDocumentSPLADEEmbedder(
            model="prithvida/SPLADE_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            parallel=1,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_to_dict(self):
        """
        Test serialization of FastembedDocumentSPLADEEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = FastembedDocumentSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder.FastembedDocumentSPLADEEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 256,
                "progress_bar": True,
                "parallel": None,
                "embedding_separator": "\n",
                "meta_fields_to_embed": [],
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedDocumentSPLADEEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = FastembedDocumentSPLADEEmbedder(
            model="prithvida/SPLADE_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            parallel=1,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder.FastembedDocumentSPLADEEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of FastembedDocumentSPLADEEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder.FastembedDocumentSPLADEEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 256,
                "progress_bar": True,
                "parallel": None,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }
        embedder = default_from_dict(FastembedDocumentSPLADEEmbedder, embedder_dict)
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 256
        assert embedder.progress_bar is True
        assert embedder.parallel is None
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of FastembedDocumentSPLADEEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder.FastembedDocumentSPLADEEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithvida/SPLADE_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }
        embedder = default_from_dict(FastembedDocumentSPLADEEmbedder, embedder_dict)
        assert embedder.model_name == "prithvida/SPLADE_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = FastembedDocumentSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name="prithvida/SPLADE_PP_en_v1", cache_dir=None, threads=None
        )

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_document_SPLADE_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = FastembedDocumentSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def _generate_mocked_sparse_embedding(self, n):
        list_of_sparse_vectors = []
        for _ in range(n):
            random_indice_length = np.random.randint(3, 15)
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
        embedder = FastembedDocumentSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: self._generate_mocked_sparse_embedding(len(x))  # noqa: ARG005

        documents = [Document(content=f"Sample-document text {i}") for i in range(5)]

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.meta["_sparse_vector"], dict)
            assert isinstance(doc.meta["_sparse_vector"]["indices"], list)
            assert isinstance(doc.meta["_sparse_vector"]["indices"][0], int)
            assert isinstance(doc.meta["_sparse_vector"]["values"], list)
            assert isinstance(doc.meta["_sparse_vector"]["values"][0], float)

    def test_embed_incorrect_input_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = FastembedDocumentSPLADEEmbedder(model="prithvida/SPLADE_PP_en_v1")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(
            TypeError,
            match="FastembedDocumentSPLADEEmbedder expects a list of Documents as input.",
        ):
            embedder.run(documents=string_input)

        with pytest.raises(
            TypeError,
            match="FastembedDocumentSPLADEEmbedder expects a list of Documents as input.",
        ):
            embedder.run(documents=list_integers_input)

    def test_embed_metadata(self):
        """
        Test for checking output dimensions and embedding dimensions for documents
        with a custom instruction and metadata.
        """
        embedder = FastembedDocumentSPLADEEmbedder(
            model="model",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="\n",
        )
        embedder.embedding_backend = MagicMock()

        documents = [Document(content=f"document-number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        embedder.run(documents=documents)

        embedder.embedding_backend.embed.assert_called_once_with(
            [
                "meta_value 0\ndocument-number 0",
                "meta_value 1\ndocument-number 1",
                "meta_value 2\ndocument-number 2",
                "meta_value 3\ndocument-number 3",
                "meta_value 4\ndocument-number 4",
            ],
            batch_size=256,
            show_progress_bar=True,
            parallel=None,
        )

    @pytest.mark.integration
    def test_run(self):
        embedder = FastembedDocumentSPLADEEmbedder(
            model="prithvida/SPLADE_PP_en_v1",
        )
        embedder.warm_up()

        doc = Document(content="Parton energy loss in QCD matter")

        result = embedder.run(documents=[doc])
        embedding = result["documents"][0].meta["_sparse_vector"]
        assert isinstance(embedding, dict)
        assert isinstance(embedding["indices"], list)
        assert isinstance(embedding["values"], list)
        assert isinstance(embedding["indices"][0], int)
        assert isinstance(embedding["values"][0], float)
