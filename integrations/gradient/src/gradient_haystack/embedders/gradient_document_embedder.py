import logging
from typing import Any, Dict, List, Optional

from gradientai import Gradient
from haystack import Document, component, default_to_dict

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
    A component for computing Document embeddings using Gradient AI API..
    The embedding of each Document is stored in the `embedding` field of the Document.

    ```python
    embedder = GradientDocumentEmbedder(
        access_token=gradient_access_token,
        workspace_id=gradient_workspace_id,
        model_name="bge_large"))
    p = Pipeline()
    p.add_component(embedder, name="document_embedder")
    p.add_component(instance=GradientDocumentEmbedder(
    p.add_component(instance=DocumentWriter(document_store=InMemoryDocumentStore()), name="document_writer")
    p.connect("document_embedder", "document_writer")
    p.run({"document_embedder": {"documents": documents}})
    ```
    """

    def __init__(
        self,
        *,
        model_name: str = "bge-large",
        batch_size: int = 32_768,
        access_token: Optional[str] = None,
        workspace_id: Optional[str] = None,
        host: Optional[str] = None,
        progress_bar: bool = True,
    ) -> None:
        """
        Create a GradientDocumentEmbedder component.

        :param model_name: The name of the model to use.
        :param batch_size: Update cycle for tqdm progress bar, default is to update every 32_768 docs.
        :param access_token: The Gradient access token. If not provided it's read from the environment
                             variable GRADIENT_ACCESS_TOKEN.
        :param workspace_id: The Gradient workspace ID. If not provided it's read from the environment
                             variable GRADIENT_WORKSPACE_ID.
        :param host: The Gradient host. By default it uses https://api.gradient.ai/.
        :param progress_bar: Whether to show a progress bar while embedding the documents.
        """
        self._batch_size = batch_size
        self._host = host
        self._model_name = model_name
        self._progress_bar = progress_bar

        self._gradient = Gradient(access_token=access_token, host=host, workspace_id=workspace_id)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self._model_name}

    def to_dict(self) -> dict:
        """
        Serialize the component to a Python dictionary.
        """
        return default_to_dict(self, workspace_id=self._gradient.workspace_id, model_name=self._model_name)

    def warm_up(self) -> None:
        """
        Load the embedding model.
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
