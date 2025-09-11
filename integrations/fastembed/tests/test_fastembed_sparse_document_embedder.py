from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack import Document, default_from_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding

from haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder import (
    FastembedSparseDocumentEmbedder,
)


class TestFastembedSparseDocumentEmbedderDoc:
    def test_init_default(self):
        """
        Test default initialization parameters for FastembedSparseDocumentEmbedder.
        """
        embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")
        assert embedder.model_name == "prithivida/Splade_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.parallel is None
        assert not embedder.local_files_only
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for FastembedSparseDocumentEmbedder.
        """
        embedder = FastembedSparseDocumentEmbedder(
            model="prithivida/Splade_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            batch_size=64,
            progress_bar=False,
            parallel=1,
            local_files_only=True,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.model_name == "prithivida/Splade_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1
        assert embedder.local_files_only
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_to_dict(self):
        """
        Test serialization of FastembedSparseDocumentEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithivida/Splade_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "batch_size": 32,
                "progress_bar": True,
                "parallel": None,
                "local_files_only": False,
                "embedding_separator": "\n",
                "meta_fields_to_embed": [],
                "model_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of FastembedSparseDocumentEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = FastembedSparseDocumentEmbedder(
            model="prithivida/Splade_PP_en_v1",
            cache_dir="fake_dir",
            threads=2,
            batch_size=64,
            progress_bar=False,
            parallel=1,
            local_files_only=True,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithivida/Splade_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
                "local_files_only": True,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "model_kwargs": None,
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of FastembedSparseDocumentEmbedder from a dictionary,
        using default initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithivida/Splade_PP_en_v1",
                "cache_dir": None,
                "threads": None,
                "batch_size": 32,
                "progress_bar": True,
                "parallel": None,
                "local_files_only": False,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }
        embedder = default_from_dict(FastembedSparseDocumentEmbedder, embedder_dict)
        assert embedder.model_name == "prithivida/Splade_PP_en_v1"
        assert embedder.cache_dir is None
        assert embedder.threads is None
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.parallel is None
        assert not embedder.local_files_only
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of FastembedSparseDocumentEmbedder from a dictionary,
        using custom initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder",  #  noqa
            "init_parameters": {
                "model": "prithivida/Splade_PP_en_v1",
                "cache_dir": "fake_dir",
                "threads": 2,
                "batch_size": 64,
                "progress_bar": False,
                "parallel": 1,
                "local_files_only": True,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }
        embedder = default_from_dict(FastembedSparseDocumentEmbedder, embedder_dict)
        assert embedder.model_name == "prithivida/Splade_PP_en_v1"
        assert embedder.cache_dir == "fake_dir"
        assert embedder.threads == 2
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.parallel == 1
        assert embedder.local_files_only
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name="prithivida/Splade_PP_en_v1",
            cache_dir=None,
            threads=None,
            local_files_only=False,
            model_kwargs=None,
        )

    @patch(
        "haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder._FastembedSparseEmbeddingBackendFactory"
    )
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")
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
        embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: self._generate_mocked_sparse_embedding(  # noqa: ARG005
            len(x)
        )

        documents = [Document(content=f"Sample-document text {i}") for i in range(5)]

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.sparse_embedding, dict)
            assert isinstance(doc.sparse_embedding["indices"], list)
            assert isinstance(doc.sparse_embedding["indices"][0], int)
            assert isinstance(doc.sparse_embedding["values"], list)
            assert isinstance(doc.sparse_embedding["values"][0], float)

    def test_embed_incorrect_input_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = FastembedSparseDocumentEmbedder(model="prithivida/Splade_PP_en_v1")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(
            TypeError,
            match=r"FastembedSparseDocumentEmbedder expects a list of Documents as input\.",
        ):
            embedder.run(documents=string_input)

        with pytest.raises(
            TypeError,
            match=r"FastembedSparseDocumentEmbedder expects a list of Documents as input\.",
        ):
            embedder.run(documents=list_integers_input)

    def test_embed_metadata(self):
        """
        Test for checking output dimensions and embedding dimensions for documents
        with a custom instruction and metadata.
        """
        embedder = FastembedSparseDocumentEmbedder(
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
            batch_size=32,
            progress_bar=True,
            parallel=None,
        )

    def test_init_with_model_kwargs_parameters(self):
        """
        Test initialization of FastembedSparseDocumentEmbedder with model_kwargs parameters.
        """
        bm25_config = {
            "k": 1.2,
            "b": 0.75,
            "avg_len": 300.0,
            "language": "english",
            "token_max_length": 50,
        }

        embedder = FastembedSparseDocumentEmbedder(
            model="Qdrant/bm25",
            model_kwargs=bm25_config,
        )

        assert embedder.model_kwargs == bm25_config

    @pytest.mark.integration
    def test_run_with_model_kwargs(self):
        """
        Integration test to check the embedding with model_kwargs parameters.
        """
        bm42_config = {
            "alpha": 0.2,
        }

        embedder = FastembedSparseDocumentEmbedder(
            model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            model_kwargs=bm42_config,
        )
        embedder.warm_up()

        doc = Document(content="Example content using BM42")

        result = embedder.run(documents=[doc])
        embedding = result["documents"][0].sparse_embedding
        embedding_dict = embedding.to_dict()

        assert isinstance(embedding, SparseEmbedding)
        assert isinstance(embedding_dict["indices"], list)
        assert isinstance(embedding_dict["values"], list)
        assert isinstance(embedding_dict["indices"][0], int)
        assert isinstance(embedding_dict["values"][0], float)

    @pytest.mark.integration
    def test_run(self):
        embedder = FastembedSparseDocumentEmbedder(
            model="prithivida/Splade_PP_en_v1",
        )
        embedder.warm_up()

        doc = Document(content="Parton energy loss in QCD matter")

        result = embedder.run(documents=[doc])
        embedding = result["documents"][0].sparse_embedding
        embedding_dict = embedding.to_dict()
        assert isinstance(embedding, SparseEmbedding)
        assert isinstance(embedding_dict["indices"], list)
        assert isinstance(embedding_dict["values"], list)
        assert isinstance(embedding_dict["indices"][0], int)
        assert isinstance(embedding_dict["values"][0], float)
