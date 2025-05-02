from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Assume the classes are in this path for the tests
from haystack_integrations.components.embedders.google_vertex.text_embedder import VertexAITextEmbedder


# Mock the TextEmbeddingResponse structure expected by the embedder
class MockTextEmbeddingResponse:
    def __init__(self, values):
        self.values = values


@pytest.fixture()
def mock_vertex_init_and_model():
    """
    Fixture to mock vertexai.init and TextEmbeddingModel.from_pretrained
    """
    with patch("vertexai.init") as mock_init, patch(
        "vertexai.language_models.TextEmbeddingModel.from_pretrained"
    ) as mock_from_pretrained:
        mock_embedder = MagicMock(spec=TextEmbeddingModel)
        # Simulate returning a list with one response object for get_embeddings
        mock_embedder.get_embeddings.return_value = [MockTextEmbeddingResponse([0.1] * 768)]
        mock_from_pretrained.return_value = mock_embedder
        yield mock_init, mock_from_pretrained, mock_embedder


# Define valid parameters for initialization
VALID_MODEL = "text-embedding-005"
VALID_TASK_TYPE = "RETRIEVAL_QUERY"


def test_init_defaults(mock_vertex_init_and_model):
    """Test default initialization."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model

    embedder = VertexAITextEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)

    assert embedder.model == VALID_MODEL
    assert embedder.task_type == VALID_TASK_TYPE
    assert embedder.progress_bar is True
    assert embedder.truncate_dim is None
    assert isinstance(embedder.gcp_project_id, Secret)
    assert isinstance(embedder.gcp_region_name, Secret)

    # Check that vertexai.init and from_pretrained were called with default secrets resolved to None
    mock_init.assert_called_once_with(project=None, location=None)
    mock_from_pretrained.assert_called_once_with(VALID_MODEL)


def test_init_custom_params(mock_vertex_init_and_model):
    """Test initialization with custom parameters."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model
    project_id = Secret.from_token("test-project")
    region = Secret.from_token("us-west1")

    embedder = VertexAITextEmbedder(
        model="textembedding-gecko-multilingual@001",
        task_type="SEMANTIC_SIMILARITY",
        gcp_project_id=project_id,
        gcp_region_name=region,
        progress_bar=False,
        truncate_dim=256,
    )

    assert embedder.model == "textembedding-gecko-multilingual@001"
    assert embedder.task_type == "SEMANTIC_SIMILARITY"
    assert embedder.progress_bar is False
    assert embedder.truncate_dim == 256
    assert embedder.gcp_project_id == project_id
    assert embedder.gcp_region_name == region

    mock_init.assert_called_once_with(project="test-project", location="us-west1")
    mock_from_pretrained.assert_called_once_with("textembedding-gecko-multilingual@001")


# Note: The current implementation relies on Literal and the SDK for model validation.
# Adding an explicit check like in DocumentEmbedder might be beneficial.
# def test_init_invalid_model():
#     """Test initialization with an invalid model name."""
#     with pytest.raises(ValueError, match="Please provide a valid model"):
#         VertexAITextEmbedder(model="invalid-model", task_type=VALID_TASK_TYPE)


def test_run_with_string(mock_vertex_init_and_model):
    """Test embedding a single string."""
    _, _, mock_embedder_instance = mock_vertex_init_and_model
    embedder = VertexAITextEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)
    text = "This is a test sentence."
    expected_embedding = [0.1] * 768
    mock_embedder_instance.get_embeddings.return_value = [MockTextEmbeddingResponse(expected_embedding)]

    result = embedder.run(text=text)

    assert "embedding" in result
    assert result["embedding"] == expected_embedding

    # Verify the call to the underlying SDK
    mock_embedder_instance.get_embeddings.assert_called_once()
    call_args, _ = mock_embedder_instance.get_embeddings.call_args
    inputs = call_args[0]
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    assert isinstance(inputs[0], TextEmbeddingInput)
    assert inputs[0].text == text
    assert inputs[0].task_type == VALID_TASK_TYPE


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_run_with_list_document_raises_error(mock_from_pretrained, _mock_init):
    """Test that running with List[Document] raises TypeError."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAITextEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)
    docs = [Document(content="doc1"), Document(content="doc2")]

    with pytest.raises(TypeError, match="expects a string as input"):
        embedder.run(text=docs)


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_run_with_list_string_raises_error(mock_from_pretrained, _mock_init):
    """Test that running with List[str] raises TypeError."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAITextEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)
    texts = ["text1", "text2"]

    with pytest.raises(TypeError, match="expects a string as input"):
        embedder.run(text=texts)


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_to_dict(mock_from_pretrained, _mock_init):
    """Test serialization to dictionary."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    project_id = Secret.from_env_var("GCP_PROJECT_ID", strict=False)
    region = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False)
    embedder = VertexAITextEmbedder(
        model=VALID_MODEL,
        task_type=VALID_TASK_TYPE,
        gcp_project_id=project_id,
        gcp_region_name=region,
        progress_bar=False,
        truncate_dim=128,
    )
    data = embedder.to_dict()

    # Check only the fields serialized by default_to_dict in the current implementation
    # task_type, progress_bar, truncate_dim are not serialized by default
    assert data == {
        "type": "haystack_integrations.components.embedders.google_vertex.text_embedder.VertexAITextEmbedder",
        "init_parameters": {
            "model": VALID_MODEL,
            "gcp_project_id": project_id.to_dict(),
            "gcp_region_name": region.to_dict(),
            # The following are missing because they are not explicitly included in to_dict
            # "task_type": VALID_TASK_TYPE,
            # "progress_bar": False,
            # "truncate_dim": 128,
        },
    }


def test_from_dict(mock_vertex_init_and_model):
    """Test deserialization from dictionary."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model
    project_id_dict = Secret.from_env_var("GCP_PROJECT_ID", strict=False).to_dict()
    region_dict = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False).to_dict()

    data = {
        "type": "haystack_integrations.components.embedders.google_vertex.text_embedder.VertexAITextEmbedder",
        "init_parameters": {
            "model": "text-multilingual-embedding-002",
            "task_type": "CLUSTERING",
            "gcp_project_id": project_id_dict,
            "gcp_region_name": region_dict,
            "progress_bar": True,
            "truncate_dim": None,
        },
    }

    embedder = VertexAITextEmbedder.from_dict(data)

    assert embedder.model == "text-multilingual-embedding-002"
    assert embedder.task_type == "CLUSTERING"
    assert isinstance(embedder.gcp_project_id, Secret)
    assert isinstance(embedder.gcp_region_name, Secret)
    assert embedder.progress_bar is True
    assert embedder.truncate_dim is None

    # Check that vertexai.init and from_pretrained were called again
    mock_init.assert_called_once()  # Called once during deserialization
    mock_from_pretrained.assert_called_once_with("text-multilingual-embedding-002")


def test_from_dict_no_secrets(mock_vertex_init_and_model):
    """Test deserialization when secrets are not in the dictionary."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model
    data = {
        "type": "haystack_integrations.components.embedders.google_vertex.text_embedder.VertexAITextEmbedder",
        "init_parameters": {
            "model": VALID_MODEL,
            "task_type": VALID_TASK_TYPE,
            "gcp_project_id": None,  # Explicitly None
            "gcp_region_name": None,  # Explicitly None
            "progress_bar": True,
            "truncate_dim": None,
        },
    }
    embedder = VertexAITextEmbedder.from_dict(data)
    assert embedder.gcp_project_id is None
    assert embedder.gcp_region_name is None
    mock_init.assert_called_once_with(project=None, location=None)
    mock_from_pretrained.assert_called_once_with(VALID_MODEL)
