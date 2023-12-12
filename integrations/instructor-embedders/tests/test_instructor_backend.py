from unittest.mock import patch


from instructor_embedders_haystack.embedding_backend.instructor_backend import _InstructorEmbeddingBackendFactory


@patch("instructor_embedders_haystack.embedding_backend.instructor_backend.INSTRUCTOR")
def test_factory_behavior(mock_instructor):  # noqa: ARG001
    embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-large", device="cpu"
    )
    same_embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend("hkunlp/instructor-large", "cpu")
    another_embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-base", device="cpu"
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend

    # restore the factory state
    _InstructorEmbeddingBackendFactory._instances = {}


@patch("instructor_embedders_haystack.embedding_backend.instructor_backend.INSTRUCTOR")
def test_model_initialization(mock_instructor):
    _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-base", device="cpu", use_auth_token="huggingface_auth_token"
    )
    mock_instructor.assert_called_once_with(
        model_name_or_path="hkunlp/instructor-base", device="cpu", use_auth_token="huggingface_auth_token"
    )
    # restore the factory state
    _InstructorEmbeddingBackendFactory._instances = {}


@patch("instructor_embedders_haystack.embedding_backend.instructor_backend.INSTRUCTOR")
def test_embedding_function_with_kwargs(mock_instructor):  # noqa: ARG001
    embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-base"
    )

    data = [["instruction", "sentence1"], ["instruction", "sentence2"]]
    embedding_backend.embed(data=data, normalize_embeddings=True)

    embedding_backend.model.encode.assert_called_once_with(data, normalize_embeddings=True)
    # restore the factory state
    _InstructorEmbeddingBackendFactory._instances = {}
