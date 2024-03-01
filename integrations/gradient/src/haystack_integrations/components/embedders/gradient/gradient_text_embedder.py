from typing import Any, Dict, List, Optional

from gradientai import Gradient
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class GradientTextEmbedder:
    """
    A component for embedding strings using models hosted on [Gradient AI](https://gradient.ai).

    Usage example:
    ```python
    embedder = GradientTextEmbedder(model="bge_large")
    p = Pipeline()
    p.add_component(instance=embedder, name="text_embedder")
    p.add_component(instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever")
    p.connect("text_embedder", "retriever")
    p.run("embed me!!!")
    ```
    """

    def __init__(
        self,
        *,
        model: str = "bge-large",
        access_token: Secret = Secret.from_env_var("GRADIENT_ACCESS_TOKEN"),  # noqa: B008
        workspace_id: Secret = Secret.from_env_var("GRADIENT_WORKSPACE_ID"),  # noqa: B008
        host: Optional[str] = None,
    ) -> None:
        """
        Create a GradientTextEmbedder component.

        :param model: The name of the model to use.
        :param access_token: The Gradient access token.
        :param workspace_id: The Gradient workspace ID.
        :param host: The Gradient host. By default, it uses https://api.gradient.ai/.
        """
        self._host = host
        self._model_name = model
        self._access_token = access_token
        self._workspace_id = workspace_id

        self._gradient = Gradient(
            host=host, access_token=access_token.resolve_value(), workspace_id=workspace_id.resolve_value()
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self._model_name}

    def to_dict(self) -> dict:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self._model_name,
            host=self._host,
            access_token=self._access_token.to_dict(),
            workspace_id=self._workspace_id.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientTextEmbedder":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["access_token", "workspace_id"])
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if not hasattr(self, "_embedding_model"):
            self._embedding_model = self._gradient.get_embeddings_model(slug=self._model_name)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Generates an embedding for a single text."""
        if not isinstance(text, str):
            msg = "GradientTextEmbedder expects a string as an input.\
                In case you want to embed a list of Documents, please use the GradientDocumentEmbedder."
            raise TypeError(msg)

        if not hasattr(self, "_embedding_model"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        result = self._embedding_model.embed(inputs=[{"input": text}])

        if (not result) or (result.embeddings is None) or (len(result.embeddings) == 0):
            msg = "The embedding model did not return any embeddings."
            raise RuntimeError(msg)

        return {"embedding": result.embeddings[0].embedding}
