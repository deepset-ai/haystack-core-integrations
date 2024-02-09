from unittest.mock import patch

from haystack_integrations.components.embedders.fastembed.embedding_backend.fastembed_backend import (
    _FastembedEmbeddingBackendFactory,
)


@patch("haystack_integrations.components.embedders.fastembed.embedding_backend.fastembed_backend.TextEmbedding")
def test_factory_behavior(mock_instructor):  # noqa: ARG001
    embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend(model_name="BAAI/bge-small-en-v1.5")
    same_embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend("BAAI/bge-small-en-v1.5")
    another_embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend(
        model_name="BAAI/bge-base-en-v1.5"
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend

    # restore the factory state
    _FastembedEmbeddingBackendFactory._instances = {}


@patch("haystack_integrations.components.embedders.fastembed.embedding_backend.fastembed_backend.TextEmbedding")
def test_model_initialization(mock_instructor):
    _FastembedEmbeddingBackendFactory.get_embedding_backend(
        model_name="BAAI/bge-small-en-v1.5",
    )
    mock_instructor.assert_called_once_with(
        model_name="BAAI/bge-small-en-v1.5",
    )
    # restore the factory state
    _FastembedEmbeddingBackendFactory._instances = {}


@patch("haystack_integrations.components.embedders.fastembed.embedding_backend.fastembed_backend.TextEmbedding")
def test_embedding_function_with_kwargs(mock_instructor):  # noqa: ARG001
    embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend(model_name="BAAI/bge-small-en-v1.5")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data)

    embedding_backend.model.embed.assert_called_once_with(data)
    # restore the factory stateTrue
    _FastembedEmbeddingBackendFactory._instances = {}
