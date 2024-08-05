import logging
from typing import Any, Dict, List, Optional

from gradientai import Gradient
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

tqdm_imported: bool = True
try:
    from tqdm import tqdm
except ImportError:
    tqdm_imported = False


logger = logging.getLogger(__name__)


def _alt_progress_bar(x: Any) -> Any:
    return x


@component
class GradientDocumentEmbedder:
    """
    A component for computing Document embeddings using Gradient AI API.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.writers import DocumentWriter
    from haystack import Document

    from haystack_integrations.components.embedders.gradient import GradientDocumentEmbedder

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=GradientDocumentEmbedder(), name="document_embedder")
    indexing_pipeline.add_component(
        instance=DocumentWriter(document_store=InMemoryDocumentStore()), name="document_writer")
    )
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})
    >>> {'document_writer': {'documents_written': 3}}
    ```
    """

    def __init__(
        self,
        *,
        model: str = "bge-large",
        batch_size: int = 32_768,
        access_token: Secret = Secret.from_env_var("GRADIENT_ACCESS_TOKEN"),  # noqa: B008
        workspace_id: Secret = Secret.from_env_var("GRADIENT_WORKSPACE_ID"),  # noqa: B008
        host: Optional[str] = None,
        progress_bar: bool = True,
    ) -> None:
        """
        Create a GradientDocumentEmbedder component.

        :param model: The name of the model to use.
        :param batch_size: Update cycle for tqdm progress bar, default is to update every 32_768 docs.
        :param access_token: The Gradient access token.
        :param workspace_id: The Gradient workspace ID.
        :param host: The Gradient host. By default, it uses [Gradient AI](https://api.gradient.ai/).
        :param progress_bar: Whether to show a progress bar while embedding the documents.
        """
        self._batch_size = batch_size
        self._host = host
        self._model_name = model
        self._progress_bar = progress_bar
        self._access_token = access_token
        self._workspace_id = workspace_id

        self._gradient = Gradient(
            access_token=access_token.resolve_value(), workspace_id=workspace_id.resolve_value(), host=host
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
            batch_size=self._batch_size,
            host=self._host,
            progress_bar=self._progress_bar,
            access_token=self._access_token.to_dict(),
            workspace_id=self._workspace_id.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientDocumentEmbedder":
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

    def _generate_embeddings(self, documents: List[Document], batch_size: int) -> List[List[float]]:
        """
        Batches the documents and generates the embeddings.
        """
        if self._progress_bar and tqdm_imported:
            batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
            progress_bar = tqdm
        else:
            # no progress bar
            progress_bar = _alt_progress_bar  # type: ignore
            batches = [documents]

        embeddings = []
        for batch in progress_bar(batches):
            response = self._embedding_model.embed(inputs=[{"input": doc.content} for doc in batch])
            embeddings.extend([e.embedding for e in response.embeddings])

        return embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.

        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        :returns:
            A dictionary with the following keys:
            - `documents`: The embedded Documents.

        """
        if not isinstance(documents, list) or documents and any(not isinstance(doc, Document) for doc in documents):
            msg = "GradientDocumentEmbedder expects a list of Documents as input.\
                  In case you want to embed a list of strings, please use the GradientTextEmbedder."
            raise TypeError(msg)

        if not hasattr(self, "_embedding_model"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        embeddings = self._generate_embeddings(documents=documents, batch_size=self._batch_size)
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        return {"documents": documents}
