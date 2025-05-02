import logging
import warnings
from typing import Any, Dict, List, Literal, Optional

from google import genai
from google.genai import types
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

# Load environment variables from .env file, if present
logger = logging.getLogger(__name__)


def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
    return secret.resolve_value() if secret else None


@component
class GoogleAIGeminiTextEmbedder:
    """
    A component for embedding text using Google AI models (e.g., Gemini).

    Usage example:
    ```python
    from haystack_integrations.components.embedders.google_ai import GoogleAIGeminiTextEmbedder
    from haystack.utils.auth import Secret

    # Ensure GOOGLE_API_KEY environment variable is set

    embedder = GoogleAIGeminiTextEmbedder(model="gemini-embedding-exp-03-07", task_type="retrieval_document")
    embedder.warm_up()

    text = "What is the meaning of life?"
    result = embedder.run([text])
    print(result['embedding'])
    print(result['meta'])

    # Example with explicit API key
    embedder_explicit_key = GoogleAIGeminiTextEmbedder(
        api_key=Secret.from_token("YOUR_GOOGLE_API_KEY"),
        model="models/embedding-001",
        task_type="retrieval_query"
    )
    embedder_explicit_key.warm_up()
    result_query = embedder_explicit_key.run(["How does quantum physics work?"])

    ```
    """

    def __init__(
        self,
        model: Literal[
            "gemini-embedding-exp-03-07", "text-embedding-004", "embedding-001"
        ] = "gemini-embedding-exp-03-07",
        api_key: Optional[Secret] = Secret.from_env_var("GEMINI_API_KEY"),  # noqa: B008
        task_type: Optional[str] = "retrieval_document",
        # Supported task types: "retrieval_query", "retrieval_document", "semantic_similarity",
        #                       "classification", "clustering", "question_answering", "fact_verification"
        # See: https://ai.google.dev/docs/embeddings#task_types
        title: Optional[str] = None,  # Relevant only for task_type="retrieval_document"
        output_dimensionality: Optional[int] = None,
    ):
        """
        Initializes the GoogleAIGeminiTextEmbedder component.

        :param model: The name of the Google AI embedding model to use.
                      Defaults to "models/embedding-001".
        :param api_key: The Google AI API key. It can be explicitly provided or automatically read from the
                        `GOOGLE_API_KEY` environment variable.
        :param task_type: The task type for the embedding model. This helps the model generate embeddings tailored to
                          the specific use case. Defaults to "retrieval_document".
        :param title: An optional title for the text, relevant only when `task_type` is "retrieval_document".
        """
        if not api_key:
            msg = (
                "GoogleAIGeminiTextEmbedder requires an API key. Set the GOOGLE_API_KEY environment variable "
                "or provide it explicitly via the api_key parameter."
            )
            raise ValueError(msg)

        self.model = model
        self.api_key = api_key
        self.task_type = task_type
        self.title = title
        self.output_dimensionality = output_dimensionality
        self._api_key_resolved: Optional[str] = None  # Store resolved key after warm_up

    def warm_up(self):
        """
        Authenticates with Google AI using the provided API key.
        """

        if self._api_key_resolved is None:
            self._api_key_resolved = resolve_secret(self.api_key)
            if not self._api_key_resolved:
                msg = "Could not resolve Google AI API key."
                raise ValueError(msg)
            try:
                self.client = genai.Client(api_key=self._api_key_resolved)
            except Exception as e:
                # Catch potential configuration errors early
                msg = f"Failed to configure Google AI client: {e}"
                raise ValueError(msg) from e
        # No specific client object to store for genai, configuration is module-level

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict() if self.api_key else None,
            task_type=self.task_type,
            title=self.title,
            output_dimensionality=self.output_dimensionality,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        # Ensure api_key is properly deserialized from Secret
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[List[float]], meta=Dict[str, Any])
    def run(self, texts: List[str]):
        """
        Embeds a list of texts using the configured Google AI model.

        :param texts: A list of strings to embed.
        :returns: A dictionary containing:
            - `embedding`: A list of embeddings, where each embedding is a list of floats.
            - `meta`: A dictionary with metadata about the operation (e.g., model name, task type).
        :raises TypeError: If the input `texts` is not a list of strings.
        :raises RuntimeError: If the embedding process fails.
        """
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            msg = "GoogleAIGeminiTextEmbedder expects a List of strings as input."
            raise TypeError(msg)
        if not texts:
            # Return empty list if no texts are provided
            return {"embedding": [], "meta": {"model": self.model, "task_type": self.task_type}}

        if self._api_key_resolved is None:
            msg = "The component has not been warmed up. Please call warm_up() before running."
            raise RuntimeError(msg)

        # Prepare parameters for the API call
        configs = types.EmbedContentConfig()
        api_params = {"model": self.model, "contents": texts, "configs": configs}

        if self.task_type:
            configs.task_type = self.task_type
        # Add title only if task_type is retrieval_document and title is provided
        if self.task_type == "retrieval_document" and self.title:
            configs.title = self.title
        elif self.title and self.task_type != "retrieval_document":
            warnings.warn(
                UserWarning("Warning: Title 'Should Be Ignored' is ignored because task_type is 'retrieval_query'"),
                stacklevel=2,
            )
        if self.output_dimensionality:
            configs.output_dimensionality = self.output_dimensionality
        try:
            # Make the API call to embed the batch of texts
            result = self.client.models.embed_content(**api_params)

        except Exception as e:
            # TODO: Add more specific error handling for common API errors if possible
            msg = f"Google AI embedding failed: {e}"
            raise RuntimeError(msg) from e

        # Extract embeddings - result.embedding should be the list of lists
        embeddings = result.get("embedding")  # Use .get for safety, returns None if key missing
        if embeddings is None:
            msg = f"Google AI API response did not contain 'embedding' key. Response: {result}"
            raise RuntimeError(msg)

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            msg = (
                f"Google AI API returned an unexpected number of embeddings "
                f"(expected {len(texts)}, got {len(embeddings)}). Response: {result}"
            )
            raise RuntimeError(msg)

        # Prepare metadata
        meta = {"model": self.model, "task_type": self.task_type}
        # Google AI API (genai) doesn't seem to consistently return usage info in embed_content response object yet.
        # If it does in the future, it could be added here. Example:
        # if usage_metadata := getattr(result, 'usage_metadata', None): # Check if attribute exists
        #    meta["usage"] = usage_metadata

        return {"embedding": embeddings, "meta": meta}
