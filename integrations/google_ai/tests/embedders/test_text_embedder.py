from unittest.mock import ANY, MagicMock, patch  # Use ANY for complex objects like configs

import pytest
from google.api_core import exceptions as google_exceptions  # Import potential exceptions
from google.genai import types  # Import types for mocking config
from haystack.utils.auth import Secret

# Assuming the file is located at:
# haystack_integrations/components/embedders/google_ai/text_embedder.py
# Adjust the path if necessary
from haystack_integrations.components.embedders.google_ai.text_embedder import GoogleAIGeminiTextEmbedder


# Mock the genai module before it's imported by the class
# We need to mock the Client and its methods
@pytest.fixture(autouse=True)
def mock_genai_client():
    # Create a mock client instance
    mock_client_instance = MagicMock()
    # Mock the embed_content method on the client's models attribute
    mock_client_instance.models.embed_content = MagicMock()

    # Patch the Client class within the text_embedder module
    with patch(
        "haystack_integrations.components.embedders.google_ai.text_embedder.genai.Client",
        return_value=mock_client_instance,
    ) as mock_client_constructor:
        yield mock_client_constructor, mock_client_instance  # Yield both constructor and instance for assertions


@pytest.fixture
def embedder(monkeypatch):
    """Creates a default GoogleAIGeminiTextEmbedder instance with a mocked API key."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    return GoogleAIGeminiTextEmbedder()


# --- Initialization Tests ---


def test_init_default_parameters(monkeypatch):
    """Test default initialization."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    embedder = GoogleAIGeminiTextEmbedder()
    assert embedder.model == "gemini-embedding-exp-03-07"  # Check default model
    assert embedder.api_key == Secret.from_env_var("GEMINI_API_KEY")
    assert embedder.task_type == "retrieval_document"
    assert embedder.title is None
    assert embedder.output_dimensionality is None
    assert embedder._api_key_resolved is None
    assert not hasattr(embedder, "client")  # Client created in warm_up


def test_init_explicit_parameters():
    """Test initialization with explicit parameters."""
    embedder = GoogleAIGeminiTextEmbedder(
        model="embedding-001",
        api_key=Secret.from_token("explicit-key"),
        task_type="retrieval_query",
        title="My Doc Title",
        output_dimensionality=256,
    )
    assert embedder.model == "embedding-001"
    assert embedder.api_key == Secret.from_token("explicit-key")
    assert embedder.task_type == "retrieval_query"
    assert embedder.title == "My Doc Title"
    assert embedder.output_dimensionality == 256
    assert embedder._api_key_resolved is None


def test_init_invalid_model_name():
    """Test that invalid model names are still accepted at init (validation might happen at API call)."""
    # The Literal type hint provides static checking, but doesn't prevent runtime assignment
    # of other strings. The API call would likely fail later.
    embedder = GoogleAIGeminiTextEmbedder(
        model="invalid-model-name",
        api_key=Secret.from_token("explicit-key"),
    )
    assert embedder.model == "invalid-model-name"


def test_init_no_api_key_raises(monkeypatch):
    """Test that ValueError is raised if API key is not provided and env var is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # Ensure env var is not set
    with pytest.raises(ValueError, match="GoogleAIGeminiTextEmbedder requires an API key"):
        # Pass api_key=None explicitly to override the default Secret.from_env_var behavior during init check
        GoogleAIGeminiTextEmbedder(api_key=None)


# --- Warm Up Tests ---


def test_warm_up_resolves_key_and_creates_client(embedder, mock_genai_client):
    """Test warm_up resolves API key and initializes the client."""
    mock_client_constructor, mock_client_instance = mock_genai_client

    embedder.warm_up()

    # Assert API key was resolved
    assert embedder._api_key_resolved == "test-api-key"
    # Assert Client constructor was called with the resolved key
    # The Client constructor should be called with the actual resolved API key.
    mock_client_constructor.assert_called_once_with(api_key="test-api-key")
    # Assert the client instance is stored
    assert embedder.client == mock_client_instance


def test_warm_up_already_warmed_up(embedder, mock_genai_client):
    """Test that warm_up doesn't re-initialize if called multiple times."""
    mock_client_constructor, _ = mock_genai_client

    embedder.warm_up()  # First call
    embedder.warm_up()  # Second call

    # Assert Client constructor was called only once
    mock_client_constructor.assert_called_once()


def test_warm_up_client_instantiation_fails(embedder, mock_genai_client):
    """Test warm_up raises ValueError if client creation fails."""
    mock_client_constructor, _ = mock_genai_client
    # Configure the mock constructor to raise an exception
    mock_client_constructor.side_effect = Exception("Client creation failed")

    with pytest.raises(ValueError, match="Failed to configure Google AI client: Client creation failed"):
        embedder.warm_up()


# --- Run Tests ---


def test_run_without_warm_up_raises(embedder):
    """Test run raises RuntimeError if warm_up wasn't called."""
    with pytest.raises(RuntimeError, match="The component has not been warmed up"):
        embedder.run(texts=["test text"])


def test_run_invalid_input_type_raises(embedder):
    """Test run raises TypeError for invalid input."""
    embedder.warm_up()  # Needs warm_up first
    with pytest.raises(TypeError, match="GoogleAIGeminiTextEmbedder expects a List of strings"):
        embedder.run(texts="not a list")  # type: ignore
    with pytest.raises(TypeError, match="GoogleAIGeminiTextEmbedder expects a List of strings"):
        embedder.run(texts=[1, 2, 3])  # type: ignore


def test_run_empty_list(embedder):
    """Test run with an empty list returns empty results."""
    embedder.warm_up()
    result = embedder.run(texts=[])
    assert result == {"embedding": [], "meta": {"model": embedder.model, "task_type": embedder.task_type}}


def test_run_api_call_success(embedder, mock_genai_client):
    """Test a successful run call."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    texts = ["text 1", "text 2"]
    expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    # Configure the mock embed_content method to return a successful response
    mock_client_instance.models.embed_content.return_value = {"embedding": expected_embeddings}

    embedder.warm_up()
    result = embedder.run(texts=texts)

    # Assert embed_content was called correctly
    mock_client_instance.models.embed_content.assert_called_once_with(
        model=embedder.model,
        contents=texts,
        configs=ANY,  # Use ANY because comparing EmbedContentConfig objects is tricky
    )
    # Check the properties of the passed config object (captured via ANY)
    call_args, call_kwargs = mock_client_instance.models.embed_content.call_args
    called_configs = call_kwargs.get("configs")
    assert isinstance(called_configs, types.EmbedContentConfig)
    assert called_configs.task_type == embedder.task_type
    assert called_configs.title is None  # Default, no title
    assert called_configs.output_dimensionality is None  # Default, no dim

    # Assert the result is correct
    assert result["embedding"] == expected_embeddings
    assert result["meta"] == {"model": embedder.model, "task_type": embedder.task_type}


def test_run_with_title_and_output_dim(monkeypatch, mock_genai_client):
    """Test run with title and output_dimensionality."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    embedder = GoogleAIGeminiTextEmbedder(
        task_type="retrieval_document",  # Title only relevant for this task type
        title="My Awesome Document",
        output_dimensionality=128,
    )
    texts = ["content of the document"]
    expected_embeddings = [[0.5, 0.6, 0.7]]
    mock_client_instance.models.embed_content.return_value = {"embedding": expected_embeddings}

    embedder.warm_up()
    result = embedder.run(texts=texts)

    # Assert embed_content was called correctly
    mock_client_instance.models.embed_content.assert_called_once_with(model=embedder.model, contents=texts, configs=ANY)
    # Check the properties of the passed config object
    call_args, call_kwargs = mock_client_instance.models.embed_content.call_args
    called_configs = call_kwargs.get("configs")
    assert isinstance(called_configs, types.EmbedContentConfig)
    assert called_configs.task_type == "retrieval_document"
    assert called_configs.title == "My Awesome Document"
    assert called_configs.output_dimensionality == 128

    assert result["embedding"] == expected_embeddings
    assert result["meta"] == {"model": embedder.model, "task_type": embedder.task_type}


def test_run_with_title_wrong_task_type(monkeypatch, mock_genai_client):
    """Test run with title but wrong task_type (should ignore title and warn)."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    embedder = GoogleAIGeminiTextEmbedder(
        task_type="retrieval_query", title="Should Be Ignored"
    )  # Title not relevant here

    texts = ["some query text"]
    expected_embeddings = [[0.8, 0.9]]
    mock_client_instance.models.embed_content.return_value = {"embedding": expected_embeddings}

    embedder.warm_up()

    with pytest.warns(
        UserWarning, match="Warning: Title 'Should Be Ignored' is ignored because task_type is 'retrieval_query'"
    ):
        result = embedder.run(texts=texts)

    # Assert embed_content was called correctly
    mock_client_instance.models.embed_content.assert_called_once_with(model=embedder.model, contents=texts, configs=ANY)
    # Check the properties of the passed config object
    call_args, call_kwargs = mock_client_instance.models.embed_content.call_args
    called_configs = call_kwargs.get("configs")
    assert isinstance(called_configs, types.EmbedContentConfig)
    assert called_configs.task_type == "retrieval_query"
    assert called_configs.title is None  # Title should NOT be set on config
    assert result["embedding"] == expected_embeddings
    assert result["meta"] == {"model": embedder.model, "task_type": embedder.task_type}


def test_run_api_error_raises(embedder, mock_genai_client):
    """Test run raises RuntimeError if the API call fails."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    # Configure the mock embed_content to raise a Google API error
    api_error = google_exceptions.InternalServerError("API failed")
    mock_client_instance.models.embed_content.side_effect = api_error

    embedder.warm_up()
    with pytest.raises(RuntimeError, match=f"Google AI embedding failed: {api_error}"):
        embedder.run(texts=["test text"])


def test_run_bad_response_no_embedding_key(embedder, mock_genai_client):
    """Test run raises RuntimeError if response lacks 'embedding' key."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    mock_client_instance.models.embed_content.return_value = {"wrong_key": []}  # Missing 'embedding'

    embedder.warm_up()
    with pytest.raises(RuntimeError, match="Google AI API response did not contain 'embedding' key"):
        embedder.run(texts=["test text"])


def test_run_bad_response_wrong_embedding_count(embedder, mock_genai_client):
    """Test run raises RuntimeError if embedding count doesn't match text count."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    texts = ["text 1", "text 2"]
    # Return only one embedding for two texts
    mock_client_instance.models.embed_content.return_value = {"embedding": [[0.1, 0.2]]}

    embedder.warm_up()
    with pytest.raises(RuntimeError, match="Google AI API returned an unexpected number of embeddings"):
        embedder.run(texts=texts)


# --- Serialization Tests ---


def test_to_dict(monkeypatch):
    """Test serialization to dictionary."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    embedder = GoogleAIGeminiTextEmbedder(
        model="embedding-001", task_type="semantic_similarity", title="Another Title", output_dimensionality=512
    )
    data = embedder.to_dict()

    assert data == {
        "type": "haystack_integrations.components.embedders.google_ai.text_embedder.GoogleAIGeminiTextEmbedder",
        "init_parameters": {
            "model": "embedding-001",
            "api_key": {"env_vars": ["GEMINI_API_KEY"], "type": "env_var", "strict": True},  # Serialized Secret
            "task_type": "semantic_similarity",
            "title": "Another Title",
            "output_dimensionality": 512,
        },
    }


def test_from_dict(monkeypatch):
    """Test deserialization from dictionary."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")  # Needed for deserializing env_var Secret
    data = {
        "type": "haystack_integrations.components.embedders.google_ai.text_embedder.GoogleAIGeminiTextEmbedder",
        "init_parameters": {
            "model": "embedding-001",
            "api_key": {"env_vars": ["GEMINI_API_KEY"], "type": "env_var", "strict": True},
            "task_type": "semantic_similarity",
            "title": "Another Title",
            "output_dimensionality": 512,
        },
    }
    embedder = GoogleAIGeminiTextEmbedder.from_dict(data)

    assert embedder.model == "embedding-001"
    # Check that the Secret object was correctly deserialized
    assert isinstance(embedder.api_key, Secret)
    assert embedder.api_key._env_vars == ("GEMINI_API_KEY",)
    assert embedder.task_type == "semantic_similarity"
    assert embedder.title == "Another Title"
    assert embedder.output_dimensionality == 512


# --- Integration Test (Serialization + Run) ---


def test_integration_to_from_dict_and_run(monkeypatch, mock_genai_client):
    """Test serialization, deserialization, and running the component."""
    mock_client_constructor, mock_client_instance = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "orig-key")  # Set env var for original instance

    # 1. Create and configure the original instance
    original_embedder = GoogleAIGeminiTextEmbedder(
        model="text-embedding-004", task_type="retrieval_document", title="Test Doc", output_dimensionality=256
    )

    # 2. Serialize it
    data = original_embedder.to_dict()

    # 3. Deserialize it (ensure env var is still set for Secret resolution)
    deserialized_embedder = GoogleAIGeminiTextEmbedder.from_dict(data)

    # 4. Assert deserialized parameters are correct
    assert deserialized_embedder.model == "text-embedding-004"
    assert deserialized_embedder.task_type == "retrieval_document"
    assert deserialized_embedder.title == "Test Doc"
    assert deserialized_embedder.output_dimensionality == 256
    # Check Secret deserialization
    assert isinstance(deserialized_embedder.api_key, Secret)
    assert deserialized_embedder.api_key._env_vars == ("GEMINI_API_KEY",)

    # 5. Warm up the deserialized instance
    # (Mock client will be used, but let's ensure the key resolution works)
    deserialized_embedder.warm_up()
    # Assert client was created (using the mock) with the resolved key
    mock_client_constructor.assert_called_with(api_key="orig-key")  # Resolved from env var
    assert deserialized_embedder.client is not None

    # 6. Prepare mock response and run
    texts = ["some document content"]
    expected_embeddings = [[0.1, 0.9, 0.2]]
    mock_client_instance.models.embed_content.return_value = {"embedding": expected_embeddings}

    result = deserialized_embedder.run(texts=texts)

    # 7. Assert the run call was successful and used correct parameters
    mock_client_instance.models.embed_content.assert_called_once_with(
        model="text-embedding-004", contents=texts, configs=ANY
    )
    call_args, call_kwargs = mock_client_instance.models.embed_content.call_args
    called_configs = call_kwargs.get("configs")
    assert called_configs.task_type == "retrieval_document"
    assert called_configs.title == "Test Doc"
    assert called_configs.output_dimensionality == 256

    assert result["embedding"] == expected_embeddings
    assert result["meta"] == {"model": "text-embedding-004", "task_type": "retrieval_document"}
