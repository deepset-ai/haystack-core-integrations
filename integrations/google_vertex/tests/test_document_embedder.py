from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses.document import Document
from haystack.utils.auth import Secret
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Assume the classes are in this path for the tests
from haystack_integrations.components.embedders.google_vertex.document_embedder import (
    VertexAIDocumentEmbedder,
)


# Mock the TextEmbeddingResponse structure expected by the embedder
class MockTextEmbeddingResponse:
    def __init__(self, values):
        self.values = values


# Mock the CountTokensResponse structure
class MockCountTokensResponse:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens


@pytest.fixture()
def mock_vertex_init_and_model():
    """
    Fixture to mock vertexai.init and TextEmbeddingModel.from_pretrained
    """
    with patch("vertexai.init") as mock_init, patch(
        "vertexai.language_models.TextEmbeddingModel.from_pretrained"
    ) as mock_from_pretrained:
        mock_embedder = MagicMock(spec=TextEmbeddingModel)
        mock_embedder.get_embeddings.return_value = [MockTextEmbeddingResponse([0.1] * 768)]
        mock_embedder.count_tokens.return_value = MockCountTokensResponse(total_tokens=10)
        mock_from_pretrained.return_value = mock_embedder
        yield mock_init, mock_from_pretrained, mock_embedder


# Define valid parameters for initialization
VALID_MODEL = "text-embedding-005"
VALID_TASK_TYPE = "RETRIEVAL_DOCUMENT"


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_init_defaults(mock_from_pretrained, mock_init):
    """Test default initialization."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAIDocumentEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)
    assert embedder.model == VALID_MODEL
    assert embedder.task_type == VALID_TASK_TYPE
    assert embedder.batch_size == 32
    assert embedder.max_tokens_total == 20000
    assert embedder.time_sleep == 30
    assert embedder.retries == 3
    assert embedder.progress_bar is True
    assert embedder.truncate_dim is None
    assert embedder.meta_fields_to_embed == []
    assert embedder.embedding_separator == "\n"
    assert isinstance(embedder.gcp_project_id, Secret)
    assert isinstance(embedder.gcp_region_name, Secret)

    mock_init.assert_called_once()
    mock_from_pretrained.assert_called_once_with(VALID_MODEL)


def test_init_custom_params(mock_vertex_init_and_model):
    """Test initialization with custom parameters."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model
    project_id = Secret.from_token("test-project")
    region = Secret.from_token("us-west1")

    embedder = VertexAIDocumentEmbedder(
        model="textembedding-gecko-multilingual@001",
        task_type="SEMANTIC_SIMILARITY",
        gcp_project_id=project_id,
        gcp_region_name=region,
        batch_size=64,
        max_tokens_total=10000,
        time_sleep=10,
        retries=5,
        progress_bar=False,
        truncate_dim=256,
        meta_fields_to_embed=["meta_key"],
        embedding_separator=" | ",
    )

    assert embedder.model == "textembedding-gecko-multilingual@001"
    assert embedder.task_type == "SEMANTIC_SIMILARITY"
    assert embedder.batch_size == 64
    assert embedder.max_tokens_total == 10000
    assert embedder.time_sleep == 10
    assert embedder.retries == 5
    assert embedder.progress_bar is False
    assert embedder.truncate_dim == 256
    assert embedder.meta_fields_to_embed == ["meta_key"]
    assert embedder.embedding_separator == " | "
    assert embedder.gcp_project_id == project_id
    assert embedder.gcp_region_name == region

    mock_init.assert_called_once_with(project="test-project", location="us-west1")
    mock_from_pretrained.assert_called_once_with("textembedding-gecko-multilingual@001")


def test_init_invalid_model():
    """Test initialization with an invalid model name."""
    with pytest.raises(ValueError, match="Please provide a valid model"):
        VertexAIDocumentEmbedder(model="invalid-model", task_type=VALID_TASK_TYPE)


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_prepare_texts_to_embed_no_meta(mock_from_pretrained, _mock_init):
    """Test _prepare_texts_to_embed without meta fields."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAIDocumentEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)
    docs = [Document(content="doc1 text"), Document(content="doc2 text")]
    texts = embedder._prepare_texts_to_embed(docs)
    assert texts == ["doc1 text", "doc2 text"]


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_prepare_texts_to_embed_with_meta(mock_from_pretrained, _mock_init):
    """Test _prepare_texts_to_embed with meta fields."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAIDocumentEmbedder(
        model=VALID_MODEL, task_type=VALID_TASK_TYPE, meta_fields_to_embed=["meta_key1", "meta_key2"]
    )
    docs = [
        Document(content="doc1 text", meta={"meta_key1": "value1"}),
        Document(content="doc2 text", meta={"meta_key1": "value2", "meta_key2": "value3"}),
        Document(content="doc3 text", meta={"other_key": "value4"}),  # meta_key1/2 missing
        Document(content=None, meta={"meta_key1": "value5"}),  # None content
    ]
    texts = embedder._prepare_texts_to_embed(docs)
    assert texts == [
        "value1\ndoc1 text",
        "value2\nvalue3\ndoc2 text",
        "doc3 text",  # Only content if specified meta keys are missing
        "value5\n",  # Separator is still added even if content is None
    ]


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_prepare_texts_to_embed_custom_separator(mock_from_pretrained, _mock_init):
    """Test _prepare_texts_to_embed with a custom separator."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAIDocumentEmbedder(
        model=VALID_MODEL, task_type=VALID_TASK_TYPE, meta_fields_to_embed=["meta_key"], embedding_separator=" --- "
    )
    docs = [Document(content="doc text", meta={"meta_key": "value"})]
    texts = embedder._prepare_texts_to_embed(docs)
    assert texts == ["value --- doc text"]


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_get_text_embedding_input(mock_from_pretrained, _mock_init):
    """Test conversion of Documents to TextEmbeddingInput."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    embedder = VertexAIDocumentEmbedder(model=VALID_MODEL, task_type="CLASSIFICATION")
    docs = [Document(content="text1"), Document(content="text2")]

    with patch.object(embedder, "_prepare_texts_to_embed", return_value=["prep_text1", "prep_text2"]) as mock_prepare:
        inputs = embedder.get_text_embedding_input(docs)

    mock_prepare.assert_called_once_with(documents=docs)
    assert len(inputs) == 2
    assert isinstance(inputs[0], TextEmbeddingInput)
    assert inputs[0].text == "prep_text1"
    assert inputs[0].task_type == "CLASSIFICATION"
    assert isinstance(inputs[1], TextEmbeddingInput)
    assert inputs[1].text == "prep_text2"
    assert inputs[1].task_type == "CLASSIFICATION"


def test_embed_batch(mock_vertex_init_and_model):
    """Test embedding a single batch successfully."""
    _, _, mock_embedder_instance = mock_vertex_init_and_model
    embedder = VertexAIDocumentEmbedder(model=VALID_MODEL, task_type=VALID_TASK_TYPE)
    docs = [Document(content="text1"), Document(content="text2")]
    prepared_texts = ["text1", "text2"]
    expected_embeddings = [[0.1] * 10, [0.2] * 10]

    # Mock the response from the underlying API
    mock_embedder_instance.get_embeddings.return_value = [
        MockTextEmbeddingResponse(expected_embeddings[0]),
        MockTextEmbeddingResponse(expected_embeddings[1]),
    ]

    with patch.object(embedder, "_prepare_texts_to_embed", return_value=prepared_texts):
        embeddings = embedder.embed_batch(docs)

    assert embeddings == expected_embeddings
    # Check that get_embeddings was called with the correct TextEmbeddingInput objects
    call_args, _ = mock_embedder_instance.get_embeddings.call_args
    inputs = call_args[0]
    assert len(inputs) == 2
    assert inputs[0].text == "text1"
    assert inputs[0].task_type == VALID_TASK_TYPE
    assert inputs[1].text == "text2"
    assert inputs[1].task_type == VALID_TASK_TYPE


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_to_dict(mock_from_pretrained, _mock_init):
    """Test serialization to dictionary."""
    mock_embedder = MagicMock(spec=TextEmbeddingModel)
    mock_from_pretrained.return_value = mock_embedder

    project_id = Secret.from_env_var("GCP_PROJECT_ID", strict=False)
    region = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False)
    embedder = VertexAIDocumentEmbedder(
        model=VALID_MODEL,
        task_type=VALID_TASK_TYPE,
        gcp_project_id=project_id,
        gcp_region_name=region,
        batch_size=64,
        progress_bar=False,
        truncate_dim=128,
        meta_fields_to_embed=["meta1"],
        embedding_separator="||",
    )
    data = embedder.to_dict()

    assert data == {
        "type": "haystack_integrations.components.embedders.google_vertex.document_embedder.VertexAIDocumentEmbedder",
        "init_parameters": {
            "model": VALID_MODEL,
            "task_type": VALID_TASK_TYPE,
            "gcp_project_id": project_id.to_dict(),
            "gcp_region_name": region.to_dict(),
            "batch_size": 64,
            "max_tokens_total": 20000,  # Default value was not overridden
            "time_sleep": 30,  # Default value was not overridden
            "retries": 3,  # Default value was not overridden
            "progress_bar": False,
            "truncate_dim": 128,
            "meta_fields_to_embed": ["meta1"],
            "embedding_separator": "||",
        },
    }


def test_from_dict(mock_vertex_init_and_model):
    """Test deserialization from dictionary."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model
    project_id_dict = Secret.from_env_var("GCP_PROJECT_ID", strict=False).to_dict()
    region_dict = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False).to_dict()

    data = {
        "type": "haystack_integrations.components.embedders.google_vertex.document_embedder.VertexAIDocumentEmbedder",
        "init_parameters": {
            "model": "text-multilingual-embedding-002",
            "task_type": "CLUSTERING",
            "gcp_project_id": project_id_dict,
            "gcp_region_name": region_dict,
            "batch_size": 16,
            "progress_bar": True,
            "truncate_dim": None,
            "meta_fields_to_embed": None,
            "embedding_separator": "\n",
            # Include defaults that might be missing if saved from older versions
            "max_tokens_total": 20000,
            "time_sleep": 30,
            "retries": 3,
        },
    }

    embedder = VertexAIDocumentEmbedder.from_dict(data)

    assert embedder.model == "text-multilingual-embedding-002"
    assert embedder.task_type == "CLUSTERING"
    assert isinstance(embedder.gcp_project_id, Secret)
    assert isinstance(embedder.gcp_region_name, Secret)
    assert embedder.batch_size == 16
    assert embedder.progress_bar is True
    assert embedder.truncate_dim is None
    assert embedder.meta_fields_to_embed == []
    assert embedder.embedding_separator == "\n"
    assert embedder.max_tokens_total == 20000
    assert embedder.time_sleep == 30
    assert embedder.retries == 3

    # Check that vertexai.init and from_pretrained were called again
    mock_init.assert_called_once()
    mock_from_pretrained.assert_called_once_with("text-multilingual-embedding-002")


def test_from_dict_no_secrets(mock_vertex_init_and_model):
    """Test deserialization when secrets are not in the dictionary."""
    mock_init, mock_from_pretrained, _ = mock_vertex_init_and_model
    data = {
        "type": "haystack_integrations.components.embedders.google_vertex.document_embedder.VertexAIDocumentEmbedder",
        "init_parameters": {
            "model": VALID_MODEL,
            "task_type": VALID_TASK_TYPE,
            "gcp_project_id": None,  # Explicitly None
            "gcp_region_name": None,  # Explicitly None
            "batch_size": 32,
            "progress_bar": True,
            "truncate_dim": None,
            "meta_fields_to_embed": None,
            "embedding_separator": "\n",
            "max_tokens_total": 20000,
            "time_sleep": 30,
            "retries": 3,
        },
    }
    embedder = VertexAIDocumentEmbedder.from_dict(data)
    assert embedder.gcp_project_id is None
    assert embedder.gcp_region_name is None
    mock_init.assert_called_once_with(project=None, location=None)
    mock_from_pretrained.assert_called_once_with(VALID_MODEL)
