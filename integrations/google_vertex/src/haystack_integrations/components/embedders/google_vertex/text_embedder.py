from typing import Any, Dict, List, Literal, Optional, Union

import vertexai
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)


@component
class VertexAITextEmbedder:
    """
    Embed text using VertexAI Text Embedder API

    Available models found here:
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#syntax

    Usage example:
    ```python
    from haystack_integrations.components.embedders.google_vertex import VertexAITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = VertexAITextEmbedder(model="text-embedding-005")

    print(text_embedder.run(text_to_embed))
    # {'embedding': [-0.08127457648515701, 0.03399784862995148, -0.05116401985287666, ...]
    ```
    """

    def __init__(
        self,
        model: Literal[
            "text-embedding-004",
            "text-embedding-005",
            "textembedding-gecko-multilingual@001",
            "text-multilingual-embedding-002",
            "text-embedding-large-exp-03-07",
        ],
        task_type: Literal[
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY",
        ] = "RETRIEVAL_QUERY",
        gcp_region_name: Optional[Secret] = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False),  # noqa: B008
        gcp_project_id: Optional[Secret] = Secret.from_env_var("GCP_PROJECT_ID", strict=False),  # noqa: B008
        progress_bar: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> None:
        """
        Initializes the TextEmbedder with the specified model, task type, and GCP configuration.

        Args:
            model (Literal["text-embedding-004", "text-embedding-005", "textembedding-gecko-multilingual@001",
                           "text-multilingual-embedding-002", "text-embedding-large-exp-03-07"]):
                The model to be used for text embedding.
            task_type (Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", "CLASSIFICATION",
                               "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION", "CODE_RETRIEVAL_QUERY"]):
                The type of task for which the embedding model will be used.
                Please refer to the VertexAI documentation for more details here:
                https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype
            gcp_region_name (Optional[Secret], optional):
                The GCP region name, fetched from environment variable "GCP_DEFAULT_REGION" if not provided.
                Defaults to None.
            gcp_project_id (Optional[Secret], optional):
                The GCP project ID, fetched from environment variable "GCP_PROJECT_ID" if not provided.
                Defaults to None.
            progress_bar (bool, optional):
                Whether to display a progress bar during operations. Defaults to True.
            truncate_dim (Optional[int], optional):
                The dimension to which embeddings should be truncated. Defaults to None.

        Returns:
            None
        """
        self.model = model
        self.progress_bar = progress_bar
        self.truncate_dim = truncate_dim

        self.gcp_project_id = gcp_project_id
        self.gcp_region_name = gcp_region_name

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        vertexai.init(project=resolve_secret(self.gcp_project_id), location=resolve_secret(self.gcp_region_name))
        self.embedder = TextEmbeddingModel.from_pretrained(self.model)
        self.task_type = task_type

    @component.output_types(embedding=List[float])
    def run(self, text: Union[List[Document], List[str], str]):
        if not isinstance(text, str):
            msg = (
                "FastembedTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the VertexAIDocumentEmbedder."
            )
            raise TypeError(msg)

        text_embed_input = [TextEmbeddingInput(text=text, task_type=self.task_type)]
        embeddings = self.embedder.get_embeddings(text_embed_input)[0].values
        return {"embedding": embeddings}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            gcp_project_id=self.gcp_project_id.to_dict() if self.gcp_project_id else None,
            gcp_region_name=self.gcp_region_name.to_dict() if self.gcp_region_name else None,
            model=self.model,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["gcp_project_id", "gcp_region_name"],
        )
        return default_from_dict(cls, data)
