from typing import Any, Dict, List, Literal, Optional, Union

import vertexai
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)


@component
class VertexAITextEmbedder:
    """
    Embed text using VertexAI Text Embeddings API.

    See available models in the official
    [Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#syntax).

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

        :param model: Name of the model to use.
        :param task_type: The type of task for which the embeddings are being generated.
                        For more information see the official [Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype).
        :param gcp_region_name: The default location to use when making API calls, if not set uses us-central-1.
        :param gcp_project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param progress_bar: Whether to display a progress bar during processing.
        :param truncate_dim: The dimension to truncate the embeddings to, if specified.
        """
        soft_deprecation_msg = (
            "This component uses a deprecated SDK. We recommend using the GoogleGenAITextEmbedder instead."
            "Documentation is available at https://docs.haystack.deepset.ai/docs/googlegenaitextembedder."
        )
        logger.warning(soft_deprecation_msg)

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
        """
        Processes text in batches while adhering to the API's token limit per request.

        :param text: The text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        """
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
